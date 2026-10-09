# Dimuon vertex BDT histograms and process overlays

These diagnostics use the saved `ShiftDimuonVertex_bdtScore` Double array.
Every existing dimuon pair is retained. There is no BDT cut and no change to
reconstruction, electronics, trigger settings or MC normalization. Generator
information enters only the optional MC diagnostic label partition.

The dedicated histogrammer writes only `dimuon_bdt/` histograms, so they can be
combined with the ordinary histogrammer output without duplicating cutflow:

```bash
source /cvmfs/sft.cern.ch/lcg/views/LCG_108/x86_64-el9-gcc13-opt/setup.sh
python3 utils/shift_bdt_histogrammer.py --input scored.root --output bdt_histograms.root
hadd combined_histograms.root ordinary_histograms.root bdt_histograms.root
```

Use new output paths. Neither tool overwrites existing diagnostics. Run ordinary
histogramming on the same scored input with its frozen configuration. The BDT
tool checks that the score exists and has Double type before reading any event,
including files with no dimuon pairs. It rejects inconsistent arrays, invalid
pair references and nonfinite/out-of-range scores. Failed outputs remain partial
and have an incomplete JSON receipt.

For the inclusive, Good and all five existing topology collections it books:

- `ShiftDimuonVertex{category}_bdtScore`: weighted score spectrum.
- The same name ending in `Genuine`, `Accidental` or `Unknown`: MC label subsets.
- Each score spectrum ending in `Unweighted`: raw pair counts.
- `ShiftDimuonVertex{category}_bdtScoreVsMass`, `...VsVz` and `...VsVertexR`:
  score versus reconstructed dimuon mass, vertex z and transverse radius.

The MC label definition is exactly the classifier's `labels.py`: distinct,
reliably hit-matched muons within 0.01 cm are genuine; duplicate muons or
separations at least 0.1 cm are accidental; uncertain matches and the tolerance
gap are unknown. DY genuine pairs remain genuine. No generator variable is
read by BDT scoring. Use `--sample-kind data` to histogram without importing or
reading truth; this mode uses unit event weight.

Simulation histograms use `shiftSamplingGenWeight` once, including its sign.
They retain Sumw2 and raw counts separately. Histograms include ROOT underflow
and overflow; an exact score of 1 is in overflow under the ordinary [0,1) ROOT
booking. Receipts verify all retained pairs and the complete weighted sum.

## Combine production bins before shape plots

A raw sum of bin histograms gives the simulated bin mixture, rather than the
expected process mixture. Prepare a JSON input manifest with schema
`shift-dimuon-bdt-bin-inputs-v1`, `complete: true`, and a `samples` list. Each
row contains `name`, `process`, `histogram_path`, and `label`; SM rows also
contain `cross_section_pb` and `full_gen_sumw` from the frozen parent GEN
authority. An optional `histogram_sha256` must match if supplied. Signal rows
have `process: signal` and distinct names for every mass/lifetime point.

```bash
python3 utils/prepare_shift_bdt_plots.py \
  --inputs bdt_bin_inputs.json --output-directory new_bdt_process_shapes
```

This applies `cross_section_pb/full_gen_sumw` to each weighted SM bin before
combining bins into QCD, DY and J/psi histograms. It preserves raw counts and
keeps signal points separate. It writes `plot_inputs.json` only as complete
after every process histogram has been written and checksummed. It never uses
selected Nano event counts or selected weight sums as the GEN denominator.

Produce the overlays with the ordinary TEA plotter:

```bash
source tea/setup.sh
export SHIFT_BDT_PLOT_INPUTS=/absolute/path/new_bdt_process_shapes/plot_inputs.json
export PYTHONPATH="$(pwd)/configs:${PYTHONPATH}"
cd bin
python plotter.py --config shift_bdt_plotter_config.py
```

`SHIFT_BDT_PLOT_OUTPUT` optionally selects a new plot directory. The config
produces PDF and PNG score shapes and reconstructed-coordinate maps for the
inclusive and Both-Both categories. The three SM process distributions are
shown as lines and normalized separately to unit area after the bin mixture
has been corrected. MC label subsets have separate overlays. These plots
diagnose classifier separation and sculpting; efficiencies and AUC belong to
the performance report. `N_eff` refers to the MC normalization fitted to data,
and is not a classifier metric. The coordinate maps are not proper-lifetime
maps; a lifetime study requires the corresponding signal MC and truth labels
for evaluation.

## Initial bounded validation, October 9

Fifteen native ROOT tests pass, covering every retained pair, signed/zero event
weights, Sumw2, MC label partitioning, missing-score zero-pair files, Double
type, failure receipts, no overwrite, correct full GEN bin scales and separate
signal point overlays. The shared CMSSW and TEA executables were not rebuilt.

The existing scored pilots produced complete supplemental histograms with
20 events and 7 J/psi pairs, 20 events and 1 DY pair, and 20 events and 1 QCD
pair. Process preparation used the actual source strata recovered from their
frozen provenance. The ordinary TEA plotter rendered the score overlays and
coordinate maps; the inclusive PNG was visually checked. These nine pairs
validate the workflow, rather than the classifier's physics performance.
The retained evidence is in workspace `validation/bdt_histograms_20261009/`;
use `pilot_process_shapes_verified/`, which carries the checked source-bin
assignments. Full merged-input results are recorded separately by the task
controller.
