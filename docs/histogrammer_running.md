<!-- Code style: human-code. -->
# Running the SHIFT MC histogrammer

## Standard and optional study configs

Histogram definitions and study controls belong in
`configs/shift_histogrammer_config.py`; plotting definitions belong in
`configs/shift_plotter_config.py` for standard campaign plots. Optional score
and event-mixing overlays use `configs/shift_bdt_mixing_plotter_config.py`. Use the
standard SHIFT application, submitter, merger and plotter when the feature
fits their ordinary histogram/plot model.
Specialized diagnostics such as ROC curves, fit checks and uncertainty plots
belong in `utils/`. Keep calculation/preparation code outside the main configs
instead of forcing every study into them.

Saved BDT score diagnostics are optional fills in the main histogrammer.
Unscored inputs retain ordinary histogramming. Set
`bdt_mixing_histogram_inventory` in `configs/shift_paths.py` to the pinned
histogram inventory for the optional study config. Its schema identifies BDT
or event-mixing plots; no separate mode switch is needed.
`utils/shift_bdt_mixing_histogram_inputs.py` verifies histogram inputs and prepares the SM bin
mixture; it does not draw plots. Standard reconstruction plots use the main config.
Full GEN normalization is verified by the existing `shift_sample_paths.py`
helper; the selected Nano count is never substituted for the parent GEN
denominator.

Event mixing constructs pairs from different events and repeats the same
pair fit; it cannot be replaced by histogram definitions alone. Its calculation
is one callable backend in `utils/shift_mixing.py`. Configure `eventMixing` and
set `runEventMixing=True` in the main histogrammer config, then use the same
`shift_histogrammer` application. The stage validates inputs and same-event fit
closure before writing mixed rows, source-event uncertainties and ROOT inputs.
Set `bdt_mixing_histogram_inventory` to its completed `mixing_plot_inputs.json`.
The optional study config reads that inventory. This study retains
its unvalidated physics/rate status; producing plots does not establish its
background-rate prediction.

Set the plotting paths in `configs/shift_paths.py`:

```python
bdt_mixing_histogram_inventory = "/absolute/path/mixing_plot_inputs.json"
bdt_mixing_plot_output = "/absolute/path/new_mixing_plots"
```

Render the completed study with the ordinary TEA plotter:

```bash
source tea/setup.sh
cd bin
python plotter.py --config shift_bdt_mixing_plotter_config.py
```

For specialized coverage and uncertainty-band checks, set
`eventMixing["diagnostics"] = True`. That optional rendering stays in `utils/`
and writes a separate diagnostic PDF/PNGs. It leaves the mixed rows, ROOT
histograms, weights and model accounting unchanged.

The superseded source/configs are byte-verified under workspace
`validation/tea_consolidation_20261012/before/`. Frozen production deployments
remain unchanged. The dated October 9 merged-BDT recovery tool reads its two
historical dependencies from the supplied prior frozen plan; future studies
use the standard configs or the optional study config as appropriate.

## BSM shapes in the standard reconstruction plots

Use `shift_plotter_config.py` for dark-photon shapes alongside the SM samples.
Choose the BSM campaign and points in `configs/shift_paths.py`, just as the SM
inputs are selected by `campaign`:

```python
bsm_campaign = "20261012-bdt-acceptance-stage1-split5"
bsm_masses_to_plot = [15, 30, 50]  # GeV
bsm_mean_lab_flights_to_plot = [None]  # Prompt points only.
```

Inputs resolve automatically to
`base_path/darkphoton_<point>/<bsm_campaign>/histograms_merged/ntuple_0.root`.
The shared helper finds the unique prepared production plan for that campaign
under workspace `validation/<task>/plan.json` or
`validation/<task>/<deployment>/plan.json`. Its seal binds the model labels,
full GEN inventory and scoring settings. The helper checks the selected
point's weights, grouping/completion receipts and histogram checksum before
returning an input.

The flight values `[None, 5, 30]` mean prompt and estimated mean lab flights of
5 and 30 metres; they are not proper decay lengths. Set either whole selector
to `None` to include all of its available values, or `[]` to include none.
For an unfinished point, a prepared histogram of its completed detector chunks
is also accepted. Merge those Nano files before running the standard
histogrammer. The prepared point receipt records the exact processed and
missing event IDs, source checksums and full parent GEN denominator. These
plot inputs live in `validation/<preparation>/<point>/point_receipt.json`;
the helper finds the matching campaign and point automatically. The canonical
production completion state stays unchanged.

A point with no available merged histogram produces a logger warning and is
omitted; plotting continues with the available points.
Unknown requested masses/lifetimes and inconsistent checksums, model labels
or provenance remain errors. Unselected unfinished points produce no warning.

Output is automatic too: `plots/SM_BSM/<bsm_campaign>/` in this repository.
Set `bsm_campaign = None` for SM-only plots. The optional
`histogram_cache_directory` stages content-verified local copies when ROOT
cannot open mounted EOS inputs; it does not change the canonical files.

Set `variable_histogram_rebin` in `shift_plotter_config.py` to merge adjacent
display bins in the irregular histograms. For example, `5` combines five bins;
use `1` to keep the existing display bins. The plotter keeps the full recorded
axis and any short final group, combines the counts and errors, then divides by the new bin
widths. SM and BSM use identical bins, and the plotted integrals are preserved.

```bash
source tea/setup.sh
cd bin
python plotter.py --config shift_plotter_config.py
```

Each BSM observable is independently matched to the total SM histogram area
using `to_background`. The bin-width normalization preserves that area for
density plots. These are signal shapes, with no BSM cross-section or rate
claim. SM cross sections and full GEN denominators retain their existing
normalization. The SM input is the frozen 187,214-event subset, with 149
original detector jobs omitted; it does not establish complete coverage.

## EOS storage layout after October 9, 2026

The production data base is `/eos/home-j/jniedzie/shift_cmssw/` (the equivalent
XRootD namespace is `/eos/user/j/jniedzie/shift_cmssw/`). It contains process-bin
directories such as `qcd_pThat0to1GeV`, `jpsi_pThat2to5GeV` and
`dy_mHat0p211317to0p5GeV`. Decimal boundaries retain every digit and use `p`.
The open upper boundary is `inf`. Each process-bin contains dated campaigns.

For example, the current QCD 0--1 detector sample is:

```text
qcd_pThat0to1GeV/shift_detector_representative_20261007_v10/
  nanoAOD/nano_job0000000.root
  nanoAOD_merged/ntuple_0.root
  histograms/histograms_job0000000.root
  histograms_merged/ntuple_0.root
  metadata/ntuple_production/job0000000/complete.json
```

Each data tier stores its regular ROOT files directly. Historical GEN, SIM,
DIGI/RAW/HLT, AODSIM and reconstruction variants remain separately identified.
Metadata preserves cross sections, normalization, sampling ledgers, original
receipts, archived configurations, logs and EOS saved versions. A legacy
campaign without a production date receives its EOS creation date in UTC;
the detector era `2023` is never treated as a production date.

Shared trigger/pileup inputs and frozen runtime/control payloads are under the
sibling `/eos/home-j/jniedzie/shift_cmssw_support/`. The migration inventory,
rename receipts and exact old-to-new path map are in workspace
`validation/storage_reorganization_20261009/`. Original production receipts
and their hashes are unchanged; `shift_storage_paths.py` resolves their old
paths only when accessing files. Histogram and plot configurations use the new
tier paths. Historical paths in dated records below describe their original
deployment, rather than current locations. Retained frozen jobs must not be
resubmitted against old paths; prepare a new deployment including the storage
resolver and the new output layout first.

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
  --condor --job_flavour longlunch --memory 6 \
  --max_materialize 100 --save_logs
```

`--dry` prepares the standard
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

For checksum-pinned runtime workers, prepare a new immutable wrapper and submit
file:

```bash
python3 utils/histogram_worker.py --source ORIGINAL_WORKER --output NEW_WORKER \
  --source-submit ORIGINAL_SUBMIT --output-submit NEW_SUBMIT
```

This writes safe startup limits and the required Condor exit policy into the
new submit file, preserving its exact input list and frozen payload. It refuses
to overwrite existing files. Runtime transfers retry
three times, with a ten-minute limit per attempt; extraction and relocation
have separate limits. Exit 75 requests another worker start, capped at four
starts. Checksum failures, analysis errors and exhausted retries remain held.
Startup timeouts and killed startup children use this retry path; a retry avoids
the previous worker node. A killed histogram application remains held.
Event-file reads, directory creation and publication use the system XRootD
tools with the Python/ROOT library environment removed from those child
processes. They retry three times before requesting a bounded restart on another
node. The application keeps its frozen Python/ROOT runtime and configuration.
Full application output is retained as gzip/base64 inside the Condor stdout
log to avoid filling the AFS quota with repeated progress messages. Decode it
with `python3 utils/histogram_worker.py --decode-log JOB.out --decoded-output
JOB.application.log`. Application failures also retain a plain stderr tail.
Use `max_materialize = 100` and `max_idle = 20`; a 1 GB runtime archive must
not be downloaded by 1,000 simultaneous cold workers. A factory limit does
not throttle jobs already materialized: recover that backlog in bounded waves.
Never overwrite a live worker or frozen analysis payload when preparing repairs.

## Merge histograms by process-bin

For raw Nano grouping before histogramming, use the existing Nano files config
and merger, rather than a separate group-merge config. Set
`nano_merge_inventory = "/absolute/path/nano_merge_inputs.json"` in
`configs/shift_paths.py`, then run:

```bash
source tea/setup.sh
cd bin
python merge.py --files_config shift_nanoAOD_files_config.py \
  --n-files-to-merge 100 --condor --dry
```

The generic JSON format has `schema: "shift-nano-merge-inputs-v1"` and a
`merge_input_groups` object mapping each destination directory to its exact
input-file list. Keep different physical signal points and process-bins in
separate groups. The config also reads a retained `inventory.json` beside its
hash-bound `histogram_inputs.txt`, verifying the original per-bin counts.
The merger rejects duplicate inputs, output/input aliases and missing files
before submission. Choose the batch size from the actual event counts; 100
twenty-event inputs yield roughly 2,000 events per grouped file. The examples
prepare a dry run; remove `--dry` only when ready to submit the reviewed plan.

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
