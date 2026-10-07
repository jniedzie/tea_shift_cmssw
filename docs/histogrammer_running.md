# Running the SHIFT MC histogrammer

The active settings now cover the representative V8 production: all six QCD
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
