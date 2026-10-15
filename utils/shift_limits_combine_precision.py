# Code style: AI-code. AI-generated verifier for a retained numerical compatibility overlay.
"""Check existing overlay evidence without building or modifying a Combine release."""
import hashlib
import json
from pathlib import Path


# The retired builder is preserved with the October 15 source-review evidence.
BUILDER_SHA256 = "9c3759c5197df8c980deb6de182b21b2d9c8ab0e6392263019098ab721eed761"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_stable_nll(config):
    source = Path(config.combine_path).resolve() / "src/CachingNLL.cc"
    library = Path(config.combine_preload_path).resolve()
    output = library.parent
    receipt = json.loads((output / "receipt.json").read_text())
    if (receipt["source_sha256"] != digest(source)
            or receipt["builder_sha256"] != BUILDER_SHA256
            or receipt["library_sha256"] != digest(library)
            or receipt["patched_source_sha256"] != digest(output / "CachingNLL.cc")):
        raise ValueError("Existing numerical overlay changed; choose a new limits_study")
    return receipt
