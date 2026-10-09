# Running the SHIFT MC histogrammer

The decay-corrected V10 campaign is controlled by DAGMan `12844748`, with
watchdog `12844749`, under `validation/ntuple_production_20261007_v10/`.
It automatically runs the existing histogrammer after each validated Nano
job and merges a bin only after all its histogram jobs succeed. The final
`final_complete.json` validates all 19 merged files and the exact event/weight
totals. The ordinary `production_complete.json` is then published for the
existing files configuration; its worker receipts are sharded into `results/gN/`.
The plot selector uses `shift_detector_representative_20261007_v10`; outputs
follow the usual `sample/campaign_bin/histograms_merged/ntuple_0.root` paths.
Detector and operational evidence is maintained in `SHIFT_RECONSTRUCTION.md`.

The preceding representative V8 production covers all six QCD
bins, six J/psi bins and seven DY mass bins. The existing files configuration
reads the existing final completion record and validated worker receipts in
`validation/ntuple_production_20261005_v8/`, refuses incomplete inputs and
assigns one unique histogram output per Nano file. It does not require an extra
canonical audit file. Missing completion records, failed receipts and mismatched
counts stop preparation with a fatal message and exit status 1. Outputs retain
the existing EOS base and `sample/campaign_bin/histograms/` convention. The previous merged
sample and native-weight lines remain commented in the existing configurations.

After reviewing the configurations, the standard parallel command is:

```bash
source tea/setup.sh
cd bin
python submitter.py --app shift_histogrammer \
  --config shift_histogrammer_config.py \
  --files_config shift_histogrammer_files_config.py \
  --condor --job_flavour longlunch --memory 2 \
  --max_materialize 1000 --save_logs
```

No histogram production has been submitted. `--dry` prepares the standard
submission files without submitting jobs. Logs are split into directories of
500 jobs; the existing runner stages EOS files locally and publishes outputs
with XRootD checksums. Application failures are returned to Condor.

The existing `weightsBranchName` setting uses `shiftSamplingGenWeight = W/p`
once. Its reader accepts both Float and Double branches. Efficiency pass/total
and truth/detector diagnostic distributions use this weight for representative
samples; restoring `genWeight` retains the previous raw diagnostics. Raw event
cutflow counts stay raw. Full parent GEN exposure/trial denominators remain in
the existing sampling plan and ledgers; selected Nano Runs/cutflow sums do not
replace them. The old plotter cross sections/selected-count normalization and
ordinary binomial efficiency errors must not be applied to this sample.

From the analysis checkout:

```bash
source tea/setup.sh
cd bin
python submitter.py --app shift_histogrammer \
  --config shift_histogrammer_config.py \
  --files_config shift_histogrammer_files_config.py --local
```

Choose the MC campaign in `configs/shift_paths.py`. The files configuration
writes to that campaign's `histograms/` directory. A direct histogrammer run
without output overrides uses the versioned local `plots/vN_<hash>/` path.
This is an MC diagnostics configuration, not an approved blinded-data workflow.

## Merged sample discovery

Both legacy `ntuple_0_<hash>.root` files and new `ntuple_sampling_complete_*`
or `ntuple_complete_*` ROOT files are supported. The new names require their
adjacent validated JSON record, matching output path and recorded SHA-256.
The recorded content hash supplies the provenance tag; the active CMSSW Git
revision is never substituted. Discovery reads metadata, not event branches;
it does not repeat the production merge's full-file checksum/physics audit.

A validated complete merge takes precedence over a legacy merge. The same
selector is used by the histogram configuration and submitter files list,
so an old partial merge and its complete replacement are not both processed.

## September 21 error repair

- The old filename-only lookup raised before the refit diagnostic histograms
  were booked. `tea` then continued with a partially executed Python config.
  Accepting the validated merge names resolves that lookup, and `ConfigManager`
  now exits nonzero on any Python configuration failure. The existing
  `VertexRefitDiagnostics_massErr` booking is retained, not silently skipped.
- A partially activated Conda environment could leave ROOT's interpreter
  unable to find `assert.h`. Before importing ROOT, the submitter restores
  `CONDA_BUILD_SYSROOT` only when absent and when the matching installed Conda
  headers exist. Explicit sysroots and non-Conda installations are unchanged.
  Sourcing `tea/setup.sh` remains the normal way to activate the complete
  runtime. ROOT diagnostics are not suppressed.
- Ten Python regression tests passed. The analysis executable/libraries were
  rebuilt and installed locally; the shared CMSSW release was untouched.
  A submitter integration run used the real configuration with zero events
  and scratch output. It selected the 200-event J/psi 1--2 bin, booked the full
  histogram set and wrote ROOT output without the reported interpreter/key
  errors. An intentionally invalid configuration exited with status 1 before
  event input/output. No real reconstructed mass values were read or plotted.

Validation logs and zero-event wrappers are in the workspace directory
`validation/histogrammer_errors_20260921/`. The existing unrelated
`TrigObj_filterBits` unsigned-64-bit reader warning is outside this repair.

## Held histogram jobs

An `OnExitHold` message records a nonzero worker exit or signal; it does not
identify the underlying error. Use `--save_logs` for submissions so stdout,
stderr and scheduler events are available in the printed submission directory.

The existing runner restores the ROOT compiler sysroot, reports application
signals and transfer failures, and replaces a job's existing output only after
the application succeeds. XRootD successful-close and checksum checks remain
required. A failed application never publishes its partial local output.
This permits rerunning the same file after a partial or overlapping submission.

For the V8 histogram run use `--job_flavour longlunch --memory 3`, as in the
parallel example above. The default `espresso` limit is only 20 minutes; CERN
can remove failed jobs under its time-limit policy before they are repaired.
The October 7 recovery uses the existing submitter and an exact missing-output
files list, without repeating the full submission.

The October 7 stalls and repeated worker failures were traced to reading Python
and ROOT libraries from the EOS-mounted TEA environment. Worker logs included
`cannot read file data: Input/output error`, including Python's `select` and
`_posixsubprocess` modules. Use an AFS environment for batch executables and
libraries; keep event inputs and histogram outputs on EOS and transfer them
with XRootD. The submitter rejects an EOS Python runtime before submission.
Moving Python alone is insufficient when the histogrammer still has EOS
library paths embedded in its executable.

The first AFS runtime passed small tests, but 1,000 simultaneous workers later
blocked in AFS before event processing. Three retained workers finished when
the other stalled workers were paused. The current recovery therefore downloads
a checksum-pinned archive through native XRootD and unpacks Python, ROOT, the
existing histogrammer and Python configurations onto each worker's local disk.
The immutable local cache is protected by a lock and reused by jobs on that
node. Embedded runtime paths in ELF binaries are also relocated; changing
library search variables alone did not remove those paths.

The archive and launcher receipts are under
`validation/histogram_stall_20261007`. Six compute-node checks covering QCD,
DY and J/psi finished in 47--56 seconds including the first cache setup;
histogramming itself took 9--16 seconds. The worker requests 6 GB disk space.
Download, extraction and relocation have bounded timeouts, and histogramming
plus publication stops after ten minutes with an informative fatal message.
Analysis settings, event weights and EOS output paths are preserved.
The submissions retain stdout, stderr and scheduler logs under
`validation/histogram_hold_20261007/afs_worker_logs`, with directories split
every 500 jobs.

## Merge histograms by process-bin

From `tea_shift_cmssw`, activate TEA and run the existing merger from `bin`:

```bash
source tea/setup.sh
cd bin
python merge.py --files_config shift_histogrammer_files_config.py \
  --condor --hadd-workers 1 --hadd-files-per-pass 100 \
  --job_flavour workday --memory 2 --dry
```

Use the same files configuration as histogram submission. For its
`input_output_file_list`, TEA merges the histogram-output column, grouped by
output directory. The Nano input column is not a merge input. Tree outputs,
when present, are grouped and merged separately. No separate merge config is
needed. Exact lists exclude unrelated files in those directories; directory
configs using `output_hists_dir` or `output_trees_dir` retain their usual behavior.

This prepares one merge job for each of the 19 process-bins, using the exact
outputs expected by the selected campaign's histogrammer files configuration. Each output
is `sample/campaign_bin/histograms_merged/ntuple_0.root`. The dry run works
while histogram jobs are running. It writes Condor files to a unique AFS
directory under `bin/tmp/condor_merge/` and prints the same concise submission
message as a real Condor run.

Once histogramming has finished, rerun the same command without `--dry` to
check that every expected input exists and is nonempty, then submit all 19
jobs. Use this command instead of submitting the dry-run `.sub` directly,
because it performs the completeness check before submission. Workers use
TEA's runtime, read EOS through XRootD, check every ROOT input, merge with the
existing `hadd` machinery, validate the local result and publish with XRootD
successful-close and checksum checks. Missing or corrupt inputs fail the job;
they are never silently skipped. Failures are held and logs stay in the printed
submission directory. Merging sums histogram weights and their stored errors;
it applies no additional normalization. Rerunning replaces the corresponding
merged output only after a successful merge.

To merge an unfinished histogram run, add `--allow-incomplete` to the same
command. It constructs a frozen list from the Nano receipts that have already
validated every detector tier, omitting missing receipts and attempts explicitly
marked unfinished or failed, with a logger warning showing the ready-job count.
Receipts claiming success still have to pass all validation checks. It then
merges the existing, nonempty histogram
files separately for each process-bin. It lists incomplete bins grouped by
process and skips bins with no available files. This option never creates or
spoofs `production_complete.json`; without it the normal command still
requires the exact complete Nano inventory. Add `--dry` as well to inspect and
prepare that partial merge. The selected input list is frozen during
preparation; files arriving later are included only when the command is rerun.
Partial merges use the same output paths, so rerun without `--allow-incomplete`
after histogramming finishes to replace them with complete results. Corrupt
ROOT files still fail the merge.
Without the option, incomplete inputs produce a Logger fatal message and exit
status 1 before any job starts, without a Python traceback.
