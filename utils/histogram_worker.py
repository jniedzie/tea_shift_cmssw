"""Prepare a new immutable worker with bounded retries for runtime startup.

This adapts the checksum-pinned local-runtime wrapper, preserving its analysis
payload and publication commands. Never overwrite a worker used by live jobs.
"""
import argparse
import base64
import gzip
from pathlib import Path
import re
import sys


ON_EXIT_HOLD = '(ExitBySignal =?= True) || ((ExitCode =!= 0) && ((ExitCode =!= 75) || (NumJobStarts >= 4)))'
ON_EXIT_REMOVE = '(ExitBySignal =?= False) && (ExitCode =?= 0)'
ON_EXIT_HOLD_REASON = '"Histogram worker failed: exit " + string(ExitCode) + "; inspect stderr for phase (runtime retry limit: 4 starts)"'
ERROR_HANDLER = '''trap 'rc=$?; echo "FATAL: Histogram worker failed during ${tea_phase} (exit ${rc}, line ${LINENO})" >&2; if [[ -n "${TEA_TRANSFER_FAILURE_MARKER:-}" && -f "${TEA_TRANSFER_FAILURE_MARKER:-}" ]]; then exit 75; fi; if [[ ( "$rc" == 124 || "$rc" == 137 ) && "$tea_phase" == runtime* ]]; then exit 75; fi; if [[ "$rc" == 75 && "$tea_phase" != runtime* ]]; then exit 1; fi; exit "$rc"' ERR'''
RUNTIME_CHECKSUM = r'''  printf '%s  %s\n' "$tea_digest" "$tea_stage/runtime.tar.gz" | sha256sum --check --status'''
RETRY_MACHINE_GUARD = '(ExitCode =!= 75) || isUndefined(LastRemoteHost) || !stringListMember(TARGET.Machine, LastRemoteHost, "@")'
LOG_BEGIN = '=== TEA APPLICATION LOG GZIP BASE64 BEGIN ==='
LOG_END = '=== TEA APPLICATION LOG GZIP BASE64 END ==='

NATIVE_XROOTD = r'''tea_native_bin="$job_sandbox/native_xrootd"
mkdir -p "$tea_native_bin"
export TEA_TRANSFER_FAILURE_MARKER="$job_sandbox/transfer_retry_exhausted"
cat > "$tea_native_bin/xrdcp" <<'TEA_NATIVE_XROOTD'
#!/bin/bash
set -eo pipefail
tea_transfer_tool="${0##*/}"
case "$tea_transfer_tool" in xrdcp) tea_transfer_seconds=120;; xrdfs) tea_transfer_seconds=60;; *) exit 1;; esac
unset PYTHONPATH PYTHONHOME LD_LIBRARY_PATH LD_PRELOAD ROOTSYS CONDA_PREFIX CONDA_BUILD_SYSROOT
export PATH=/usr/bin:/bin
printf '%s\n' "$tea_transfer_tool" > "$TEA_TRANSFER_FAILURE_MARKER"
for tea_transfer_attempt in 1 2 3; do
  if timeout -k 15 "$tea_transfer_seconds" "/usr/bin/$tea_transfer_tool" "$@"; then
    rm -f -- "$TEA_TRANSFER_FAILURE_MARKER"
    exit 0
  else
    tea_transfer_rc=$?
    echo "Native $tea_transfer_tool failed (exit $tea_transfer_rc), attempt $tea_transfer_attempt/3" >&2
  fi
  if [[ "$tea_transfer_attempt" -lt 3 ]]; then sleep $((2 * tea_transfer_attempt)); fi
done
echo "RETRYABLE: Native $tea_transfer_tool exhausted its three attempts" >&2
exit 75
TEA_NATIVE_XROOTD
cp "$tea_native_bin/xrdcp" "$tea_native_bin/xrdfs"
chmod 700 "$tea_native_bin/xrdcp" "$tea_native_bin/xrdfs"
'''


def capture_application(command):
    return f'''tea_application_log="$job_sandbox/histogram_application.log"
if {command} > "$tea_application_log" 2>&1; then
  tea_application_rc=0
else
  tea_application_rc=$?
fi
/usr/bin/gzip -c "$tea_application_log" > "$tea_application_log.gz"
echo "{LOG_BEGIN}"
/usr/bin/base64 "$tea_application_log.gz"
echo "{LOG_END}"
echo "Histogram application exited $tea_application_rc; full log retained above in gzip/base64 form"
if [[ "$tea_application_rc" != 0 ]]; then
  /usr/bin/tail -c 8192 "$tea_application_log" >&2
  echo "FATAL: Histogram application or publication failed (exit $tea_application_rc)" >&2
  if [[ -n "${{TEA_TRANSFER_FAILURE_MARKER:-}}" && -f "${{TEA_TRANSFER_FAILURE_MARKER:-}}" ]]; then exit 75; fi
  if [[ "$tea_application_rc" == 75 ]]; then exit 1; fi
  exit "$tea_application_rc"
fi'''


def decode_application_log(text):
    if text.count(LOG_BEGIN) != 1 or text.count(LOG_END) != 1:
        raise ValueError('A complete, unique compressed application log is required')
    encoded = text.split(LOG_BEGIN, 1)[1].split(LOG_END, 1)[0]
    return gzip.decompress(base64.b64decode(''.join(encoded.split()), validate=True))

DOWNLOAD = r'''  # Retry only runtime transfer failures. Analysis failures remain fatal.
  tea_download_ok=false
  for tea_attempt in 1 2 3; do
    echo "Runtime download attempt ${tea_attempt}/3 at $(date --iso-8601=seconds)"
    if timeout -k 15 600 /usr/bin/xrdcp --silent --force --cksum adler32 "$tea_url" "$tea_stage/runtime.tar.gz"; then
      tea_download_ok=true
      break
    else
      tea_transfer_rc=$?
      echo "Runtime transfer failed (exit ${tea_transfer_rc}); attempt ${tea_attempt}/3" >&2
    fi
    if [[ "$tea_attempt" -lt 3 ]]; then sleep $((30 * tea_attempt)); fi
  done
  if [[ "$tea_download_ok" != true ]]; then
    echo "RETRYABLE: Runtime download exhausted its three attempts" >&2
    exit 75
  fi'''


def harden_worker(source):
    if source.count(RUNTIME_CHECKSUM) != 1:
        raise ValueError('Unsupported worker: a pinned runtime SHA-256 check is required')
    replacements = {
        '''trap 'rc=$?; echo "FATAL: Histogram worker failed during ${tea_phase} (exit ${rc}, line ${LINENO})" >&2; exit "$rc"' ERR''':
        ERROR_HANDLER,
        'timeout -k 15 620 flock -w 600 9':
        '''if ! timeout -k 15 2720 flock -w 2700 9; then
  echo "RETRYABLE: Runtime cache lock timed out" >&2
  exit 75
fi''',
        '  timeout -k 15 240 /usr/bin/xrdcp --silent --cksum adler32 "$tea_url" "$tea_stage/runtime.tar.gz"': DOWNLOAD,
        'timeout -k 15 180 tar -xzf': 'timeout -k 15 600 tar -xzf',
        'timeout -k 15 120 /usr/bin/python3 - "$tea_stage"':
        'timeout -k 15 300 /usr/bin/python3 - "$tea_stage"',
    }
    for old, new in replacements.items():
        if source.count(old) != 1:
            raise ValueError(f'Unsupported worker: expected exactly one {old!r}')
        source = source.replace(old, new)
    runtime_path = 'export PATH="$tea_cache/runtime/bin:/usr/bin:/bin"'
    if source.count(runtime_path) != 1:
        raise ValueError('Unsupported worker: exactly one runtime activation is required')
    source = source.replace(runtime_path, NATIVE_XROOTD + 'export PATH="$tea_native_bin:$tea_cache/runtime/bin:/usr/bin:/bin"')
    commands = re.findall(r'^timeout -k 15 600 [^\n]*condor_runner\.py \\\n(?:[^\n]*\\\n)*[^\n]*', source, re.MULTILINE)
    if len(commands) != 1:
        raise ValueError('Unsupported worker: exactly one histogram application is required')
    source = source.replace(commands[0], capture_application(commands[0].replace('timeout -k 15 600', 'timeout -k 15 1200')))
    return source


def harden_submit(source, worker):
    """Preserve the inventory/payload, replacing only worker and exit policy."""
    settings = {
        'executable': str(worker.resolve()),
        'request_memory': '6000',
        'max_materialize': '100',
        'max_idle': '20',
        'on_exit_hold': ON_EXIT_HOLD,
        'on_exit_remove': ON_EXIT_REMOVE,
        'on_exit_hold_reason': ON_EXIT_HOLD_REASON,
        'periodic_release': 'False',
        'periodic_remove': 'False',
    }
    lines = source.splitlines()
    if sum(bool(re.match(r'\s*queue\b', line, re.IGNORECASE)) for line in lines) != 1:
        raise ValueError('Unsupported submit file: exactly one Queue inventory is required')
    requirements = [line.split('=', 1)[1].strip() for line in lines if re.match(r'\s*requirements\s*=', line, re.IGNORECASE)]
    if len(requirements) > 1:
        raise ValueError('Duplicate submit setting: requirements')
    settings['requirements'] = '(' + RETRY_MACHINE_GUARD + ')'
    if requirements:
        settings['requirements'] = '(' + requirements[0] + ') && ' + settings['requirements']
    for key, value in settings.items():
        matches = [i for i, line in enumerate(lines) if re.match(rf'\s*{key}\s*=', line, re.IGNORECASE)]
        if len(matches) > 1:
            raise ValueError(f'Duplicate submit setting: {key}')
        if matches:
            lines[matches[0]] = f'{key} = {value}'
        else:
            # Put new settings before Queue so they apply to the whole factory.
            queue = next((i for i, line in enumerate(lines) if re.match(r'\s*queue\b', line, re.IGNORECASE)), None)
            if queue is None:
                raise ValueError('Unsupported submit file: Queue is required')
            lines.insert(queue, f'{key} = {value}')
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--source-submit', type=Path)
    parser.add_argument('--output-submit', type=Path)
    parser.add_argument('--decode-log', type=Path)
    parser.add_argument('--decoded-output', type=Path)
    args = parser.parse_args()
    if args.decode_log:
        data = decode_application_log(args.decode_log.read_text())
        if args.decoded_output:
            with args.decoded_output.open('xb') as output:
                output.write(data)
        else:
            sys.stdout.buffer.write(data)
        return
    if not args.source or not args.output:
        parser.error('--source and --output are required when preparing a worker')
    if bool(args.source_submit) != bool(args.output_submit):
        parser.error('--source-submit and --output-submit must be used together')
    text = harden_worker(args.source.read_text())
    submit = harden_submit(args.source_submit.read_text(), args.output) if args.source_submit else None
    for destination in [args.output, *([args.output_submit] if submit else [])]:
        if destination.exists():
            raise FileExistsError(f'Refusing to overwrite existing worker/submit file: {destination}')
    with args.output.open('x') as output:
        output.write(text)
    args.output.chmod(0o755)
    if submit:
        with args.output_submit.open('x') as output:
            output.write(submit)
    print('Use max_materialize = 100 and max_idle = 20 for the runtime factory.')
    print('on_exit_hold = ' + ON_EXIT_HOLD)
    print('on_exit_remove = ' + ON_EXIT_REMOVE)
    print('on_exit_hold_reason = ' + ON_EXIT_HOLD_REASON)


if __name__ == '__main__':
    main()
