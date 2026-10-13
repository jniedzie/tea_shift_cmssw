# Code style: AI-code. AI-generated storage validation helper.
"""Build the dated storage layout and resolve preserved production paths.

Migration changes file locations, never the immutable production receipts. An
audited, completed path map connects those receipts to the relocated payloads.
An independent retention overlay records archive moves and exact retired paths.
"""
from datetime import datetime
from decimal import Decimal
from functools import lru_cache
import json
import os
from pathlib import Path, PurePosixPath
import re
from urllib.parse import urlsplit


DEFAULT_MIGRATION_MANIFEST = (
    Path(__file__).resolve().parents[2]
    / "validation/storage_reorganization_20261009/path_map.json"
)
DEFAULT_RETENTION_MANIFEST = (
    Path(__file__).resolve().parents[2]
    / "validation/pilot_campaign_cleanup_20261009/retention_path_map.json"
)
_LOCAL_PREFIX = "/eos/home-j/jniedzie"
_CANONICAL_PREFIX = "/eos/user/j/jniedzie"
_SHIFT_ROOTS = (_CANONICAL_PREFIX + "/shift_cmssw",
                _CANONICAL_PREFIX + "/shift_cmssw_support")


def _is_shift_path(path):
    return any(path == root or path.startswith(root + "/") for root in _SHIFT_ROOTS)


def _decimal_label(value):
    value = str(value).replace("p", ".")
    if not re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", value):
        raise ValueError(f"Invalid nonnegative bin boundary: {value}")
    # Decimal preserves all supplied digits, including the low-mass DY edge.
    text = format(Decimal(value), "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    text = text.lstrip("0") or "0"
    if text.startswith("."):
        text = "0" + text
    return text.replace(".", "p")


def normalize_bin(bin_name):
    """Encode decimal points as p and the open upper bound as inf."""
    bounds = str(bin_name).split("to")
    if len(bounds) != 2:
        raise ValueError(f"Expected LOWtoHIGH bin: {bin_name}")
    low, high = bounds
    low_label = _decimal_label(low)
    if high in ("inf", "-1"):
        return low_label + "toinf"
    high_label = _decimal_label(high)
    if Decimal(low_label.replace("p", ".")) >= Decimal(high_label.replace("p", ".")):
        raise ValueError(f"Bin upper boundary must exceed its lower boundary: {bin_name}")
    return low_label + "to" + high_label


def process_bin_directory(process, bin_name):
    """Return a process/bin name, retaining the production bin definition."""
    if process not in ("qcd", "jpsi", "dy"):
        raise ValueError(f"Unknown binned background process: {process}")
    variable = "mHat" if process == "dy" else "pThat"
    return f"{process}_{variable}{normalize_bin(bin_name)}GeV"


def campaign_directory(base_path, process, bin_name, campaign):
    """Require a real production date, independently of the detector year."""
    if not re.fullmatch(r"[A-Za-z0-9_-]+", campaign):
        raise ValueError(f"Invalid campaign directory name: {campaign}")
    dates = re.findall(r"(?<![0-9])(20[0-9]{6})(?![0-9])", campaign)
    if not dates:
        raise ValueError(f"Campaign name must contain its YYYYMMDD production date: {campaign}")
    for date in dates:
        try:
            datetime.strptime(date, "%Y%m%d")
        except ValueError as exc:
            raise ValueError(f"Campaign contains an invalid production date: {date}") from exc
    return Path(base_path) / process_bin_directory(process, bin_name) / campaign


def tier_directory(base_path, process, bin_name, campaign, tier):
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", tier):
        raise ValueError(f"Invalid data tier: {tier}")
    return campaign_directory(base_path, process, bin_name, campaign) / tier


def _canonical_path(path):
    path = str(path)
    if ".." in PurePosixPath(path).parts or not path.startswith("/"):
        raise ValueError(f"Storage mapping requires an absolute path without traversal: {path}")
    path = str(PurePosixPath(path))
    if path == _LOCAL_PREFIX or path.startswith(_LOCAL_PREFIX + "/"):
        path = _CANONICAL_PREFIX + path[len(_LOCAL_PREFIX):]
    return path


def _unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate storage mapping key: {key}")
        result[key] = value
    return result


@lru_cache(maxsize=4)
def _read_map(filename, mtime_ns, size):
    del mtime_ns, size  # Cache key detects a replacement after first discovery.
    record = json.loads(Path(filename).read_text(), object_pairs_hook=_unique_keys)
    if record.get("schema") != "shift-storage-path-map-v1" or record.get("complete") is not True:
        raise RuntimeError(f"Storage migration is not validated and complete: {filename}")
    paths = record.get("paths")
    aliases = record.get("aliases", {})
    directories = record.get("directories", {})
    prefixes = record.get("support_prefixes", {})
    if any(not isinstance(mapping, dict) for mapping in (paths, aliases, directories, prefixes)):
        raise RuntimeError(f"Invalid storage migration mapping: {filename}")
    result = {}
    for section, mapping in (("paths", paths), ("aliases", aliases),
                             ("directories", directories), ("support_prefixes", prefixes)):
        normalized = {}
        for old, new in mapping.items():
            if not isinstance(new, str):
                raise RuntimeError(f"Invalid storage mapping destination: {old}")
            key, target = _canonical_path(old), _canonical_path(new)
            if key in normalized and normalized[key] != target:
                raise RuntimeError(f"Conflicting EOS alias mappings: {old}")
            normalized[key] = target
        result[section] = normalized
    return result


def load_migration_map(migration_manifest=None):
    """Load a completed map; an absent default keeps pre-migration workflows valid."""
    configured = migration_manifest or os.environ.get("SHIFT_STORAGE_MIGRATION_MANIFEST")
    manifest = Path(configured or DEFAULT_MIGRATION_MANIFEST)
    try:
        info = manifest.stat()
    except FileNotFoundError:
        if configured:
            raise RuntimeError(f"Configured storage migration map is missing: {manifest}")
        return None
    return _read_map(str(manifest.resolve()), info.st_mtime_ns, info.st_size)


@lru_cache(maxsize=4)
def _read_retention_map(filename, mtime_ns, size):
    del mtime_ns, size
    record = json.loads(Path(filename).read_text(), object_pairs_hook=_unique_keys)
    if (not isinstance(record, dict) or
            record.get("schema") != "shift-storage-retention-map-v1" or
            record.get("complete") is not True):
        raise RuntimeError(f"Storage retention is not validated and complete: {filename}")
    prefixes, deleted = record.get("archive_prefixes"), record.get("deleted_paths")
    if not isinstance(prefixes, dict) or not isinstance(deleted, list):
        raise RuntimeError(f"Invalid storage retention mapping: {filename}")
    normalized = {}
    archive_root = _CANONICAL_PREFIX + "/shift_cmssw_support/legacy_references/"
    for old, new in prefixes.items():
        if not isinstance(old, str) or not isinstance(new, str):
            raise RuntimeError(f"Invalid storage retention prefix: {old}")
        key, target = _canonical_path(old), _canonical_path(new)
        if not _is_shift_path(key) or not target.startswith(archive_root):
            raise ValueError(f"Storage retention prefix is outside its SHIFT archive scope: {old}")
        if key in normalized:
            raise ValueError(f"Duplicate canonical storage retention prefix: {old}")
        normalized[key] = target
    tombstones = set()
    for path in deleted:
        if not isinstance(path, str):
            raise RuntimeError(f"Invalid retired storage path: {path}")
        key = _canonical_path(path)
        if not _is_shift_path(key):
            raise ValueError(f"Retired storage path is outside SHIFT EOS: {path}")
        if key in tombstones:
            raise ValueError(f"Duplicate retired storage path: {path}")
        tombstones.add(key)
    return {"archive_prefixes": normalized, "deleted_paths": frozenset(tombstones)}


def load_retention_map(retention_manifest=None):
    """Load an independently validated overlay without rewriting the migration."""
    manifest = Path(retention_manifest or DEFAULT_RETENTION_MANIFEST)
    try:
        info = manifest.stat()
    except FileNotFoundError:
        if retention_manifest is not None:
            raise RuntimeError(f"Configured storage retention map is missing: {manifest}")
        return None
    return _read_retention_map(str(manifest.resolve()), info.st_mtime_ns, info.st_size)


def resolve_storage_path(path, migration_manifest=None, retention_manifest=None):
    """Resolve migration then retention, preserving local EOS spelling or PFN.

    Explicit migration maps, including environment-selected maps, opt out of
    default retention. Pass retention_manifest to opt in for a custom map.
    """
    original = str(path)
    remote = urlsplit(original) if original.startswith("root://") else None
    raw = remote.path.lstrip("/") if remote else original
    raw = "/" + raw if remote else raw
    canonical = _canonical_path(raw)
    configured = migration_manifest or os.environ.get("SHIFT_STORAGE_MIGRATION_MANIFEST")
    if not configured and not _is_shift_path(canonical):
        # The default migration is scoped to SHIFT EOS storage. A guard during
        # relocation must not block unrelated local manifests or scratch files.
        return original
    mapping = load_migration_map(migration_manifest)
    target = None
    if mapping is not None:
        target = mapping["paths"].get(canonical)
        if target is None:
            target = mapping["aliases"].get(canonical)
        if target is None:
            target = mapping["directories"].get(canonical)
        if target is None:
            for prefix in sorted(mapping["support_prefixes"], key=len, reverse=True):
                if canonical == prefix or canonical.startswith(prefix + "/"):
                    target = mapping["support_prefixes"][prefix] + canonical[len(prefix):]
                    break
    migrated = target or canonical
    if ((retention_manifest is not None or not configured) and
            (_is_shift_path(canonical) or _is_shift_path(migrated))):
        retention = load_retention_map(retention_manifest)
        if retention is not None:
            if (canonical in retention["deleted_paths"] or
                    migrated in retention["deleted_paths"]):
                raise FileNotFoundError(f"SHIFT data was retired by validated cleanup: {canonical}")
            for prefix in sorted(retention["archive_prefixes"], key=len, reverse=True):
                if migrated == prefix or migrated.startswith(prefix + "/"):
                    target = retention["archive_prefixes"][prefix] + migrated[len(prefix):]
                    break
    if target is None:
        return original
    if remote:
        if remote.query or remote.fragment:
            raise ValueError("Storage PFNs with query strings or fragments are unsupported")
        return "root://" + remote.netloc + "/" + target
    if raw == _LOCAL_PREFIX or raw.startswith(_LOCAL_PREFIX + "/"):
        if target == _CANONICAL_PREFIX or target.startswith(_CANONICAL_PREFIX + "/"):
            return _LOCAL_PREFIX + target[len(_CANONICAL_PREFIX):]
    return target
