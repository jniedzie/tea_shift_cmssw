#!/bin/bash
# Code style: AI-code. AI-generated frozen-runtime activation for batch workers.
# Source with: source activate_streaming_runtime.sh ARCHIVE_ROOT ANALYSIS_OVERLAY
# The 8aaaf364f1df archive contains runtime/, bin/, prefixes.py and
# prefix_records.json. ANALYSIS_OVERLAY contains bin/, configs/, utils/ and
# source_manifest.json. Both archives must be SHA-256 checked before extraction.
# The caller's EXIT cleanup must remove SHIFT_TEA_ALIAS_DIRECTORY, if set.

stream_archive_root="${1:?Pass the extracted TEA runtime archive directory}"
stream_overlay="${2:?Pass the extracted analysis overlay directory}"
[[ -f "$stream_archive_root/prefix_records.json" && -f "$stream_overlay/source_manifest.json" ]] || return 1

# Binary conda prefixes have a fixed maximum length. A short private alias
# works even when Condor's scratch directory has a long name.
export SHIFT_TEA_ALIAS_DIRECTORY
SHIFT_TEA_ALIAS_DIRECTORY=$(mktemp -d "/tmp/stea_${UID}_XXXXXX") || return 1
ln -s "$stream_archive_root/runtime" "$SHIFT_TEA_ALIAS_DIRECTORY/r" || return 1
stream_runtime="$SHIFT_TEA_ALIAS_DIRECTORY/r"
env -u PYTHONHOME -u PYTHONPATH -u LD_LIBRARY_PATH -u LD_PRELOAD /usr/bin/python3 - \
  "$stream_archive_root" "$stream_overlay" "$stream_runtime" <<'PY'
import hashlib, json, sys
from pathlib import Path
archive, overlay, runtime = map(Path, sys.argv[1:])
sys.path.insert(0, str(archive))
from prefixes import update_prefix
manifest = json.loads((overlay / 'source_manifest.json').read_text())
for name, row in manifest.items():
    if name == 'relocation_prefixes':
        continue
    path = overlay / name
    if path.stat().st_size != row['bytes'] or hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
        raise RuntimeError('Frozen analysis payload changed: ' + name)
for target, old, mode in json.loads((archive / 'prefix_records.json').read_text()):
    if mode == 'binary' and len(str(runtime)) > len(old):
        raise RuntimeError('Runtime alias is too long for binary relocation')
    update_prefix(str(archive / target), str(runtime), old, mode)
prefixes = manifest['relocation_prefixes']
for name in ('shift_histogrammer', 'libcore.so', 'libextensions.so', 'libhistogramming.so'):
    path = str(overlay / 'bin' / name)
    update_prefix(path, str(runtime), prefixes['runtime'], 'binary')
    update_prefix(path, str(overlay / 'bin'), prefixes['analysis_bin'], 'binary')
PY
stream_activation_status=$?
if [[ "$stream_activation_status" != 0 ]]; then
  rm -rf -- "$SHIFT_TEA_ALIAS_DIRECTORY"
  unset SHIFT_TEA_ALIAS_DIRECTORY
  return "$stream_activation_status"
fi

unset PYTHONPATH PYTHONHOME LD_LIBRARY_PATH LD_PRELOAD ROOTSYS
export PATH="$stream_runtime/bin:/usr/bin:/bin"
export PYTHONHOME="$stream_runtime" CONDA_PREFIX="$stream_runtime" ROOTSYS="$stream_runtime"
export CONDA_BUILD_SYSROOT="$stream_runtime/x86_64-conda-linux-gnu/sysroot"
export LD_LIBRARY_PATH="$stream_overlay/bin:$stream_runtime/lib:$stream_runtime/lib/python3.12/site-packages/correctionlib/lib"
export PYTHONPATH="$stream_overlay/configs:$stream_overlay/bin"
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
timeout -k 10 30 /usr/bin/ldd "$stream_overlay/bin/shift_histogrammer" > "$stream_archive_root/library_paths.txt" || return 1
if /usr/bin/grep -E 'not found|/eos/|/afs/' "$stream_archive_root/library_paths.txt"; then
  echo "Frozen TEA worker still uses shared filesystem libraries" >&2
  return 1
fi
