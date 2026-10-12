# Dimuon vertex BDT histograms and process overlays

These diagnostics use the saved `ShiftDimuonVertex_bdtScore` Double array.
Every existing dimuon pair is retained. There is no BDT cut and no change to
reconstruction, electronics, trigger settings or MC normalization. Generator
information enters only the optional MC diagnostic label partition.

The ordinary SHIFT histogrammer books and fills `dimuon_bdt/` alongside the
existing diagnostics and cutflow. Use the main configuration and runner:

```bash
source tea/setup.sh
cd bin
./shift_histogrammer --config shift_histogrammer_config.py \
  --input_path scored.root --output_hists_path histograms.root
```

Use a new output path. `enableBdtDiagnostics=True` enables this optional part
of the main configuration. Unscored Nano files retain ordinary histogramming;
set `requireBdtScore=True` when a scored input is required. Present scores must
be native Double arrays counted by `nShiftDimuonVertex`, including empty input
files. Missing diagnostic branches, invalid pair references and nonfinite or
out-of-range scores fail the application before histograms are saved. Existing
frozen workers retain their original independent payloads.

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
read by BDT scoring. `enableTruthDiagnostics=False` disables all truth-based
fills, including the BDT labels, and uses unit event weight. The additional
`bdtTruthDiagnostics=False` disables only the score label partition.

Simulation histograms use `shiftSamplingGenWeight` once, including its sign.
They retain Sumw2 and raw counts separately. Histograms include ROOT underflow
and overflow; an exact score of 1 is in overflow under the ordinary [0,1) ROOT
booking. `ShiftDimuonVertex_bdtHistogramMetadata` records score availability,
event/pair counts, signed weight sums and squared-weight sums, label counts and
nonfinite map coordinates. Saved input `ShiftDimuonVertex_bdtScoreMetadata`
is copied unchanged into the ordinary output. Custom BDT fills retain Double
event-weight precision without applying a second sampling correction.

Histogram metadata describes one input file. A histogram merge sums histogram
contents and Sumw2; its retained scalar metadata is not a merged event-count
authority. Use the exact input manifest and merged histogram accounting.
Diagnostic metadata has `scope='input_file'`. `hadd` retains scalar metadata
from an input instead of summing its counters; read merged counts and weight
sums from the histograms and exact input manifest.

## Combine production bins before shape plots

A raw sum of bin histograms gives the simulated bin mixture, rather than the
expected process mixture. Prepare a JSON input manifest with schema
`shift-dimuon-bdt-bin-inputs-v1`, `complete: true`, and a `samples` list (the
histogram merger's equivalent `rows` list is also accepted). Each
row contains `name`, `process`, `histogram_path`, and `label`; SM rows also
contain `cross_section_pb` and `full_gen_sumw` from the frozen parent GEN
authority. An optional `histogram_sha256` must match if supplied. Signal rows
have `process: signal` and distinct names for every mass/lifetime point.

The main `shift_plotter_config.py` reads this inventory and applies
`cross_section_pb/full_gen_sumw` to each weighted SM bin before
combining bins into QCD, DY and J/psi histograms. It preserves raw counts and
keeps signal points separate. It writes `plot_inputs.json` only as complete
after every process histogram has been written and checksummed. It never uses
selected Nano event counts or selected weight sums as the GEN denominator.

Prepare the process mixture and render it in the ordinary TEA plotter:

```bash
source tea/setup.sh
export SHIFT_PLOT_INPUTS=/absolute/path/bdt_bin_inputs.json
export SHIFT_PLOT_MODE=bdt
export SHIFT_PLOT_PREPARATION_DIR=/absolute/path/new_process_shapes
export SHIFT_PLOT_OUTPUT=/absolute/path/new_plots
export PYTHONPATH="$(pwd)/configs:${PYTHONPATH}"
cd bin
python plotter.py --config shift_plotter_config.py
```

`SHIFT_PLOT_OUTPUT` selects the plot directory. The config
produces PDF and PNG score shapes, plus PDF reconstructed-coordinate maps for the
inclusive and Both-Both categories. The three SM process distributions are
shown as lines and normalized separately to unit area after the bin mixture
has been corrected. MC label subsets have separate overlays. These plots
diagnose classifier separation and sculpting; efficiencies and AUC belong to
the performance report. `N_eff` refers to the MC normalization fitted to data,
and is not a classifier metric. The coordinate maps are not proper-lifetime
maps; a lifetime study requires the corresponding signal MC and truth labels
for evaluation.

Set `SHIFT_PLOT_SCOPE=sm` for the three SM process overlays, or `all`
(the default) to include each signal point. Use distinct
`SHIFT_PLOT_OUTPUT` directories when rendering both. The all-sample view
uses an external legend when there are more than five curves and distinct
signal colors/styles. SM process legends are QCD, DY and J/psi after combining
their bins. The main config resolves either equivalent EOS user mount while
retaining the original input path and checksum in its receipt.

The completed process manifest can also be supplied directly through
`SHIFT_PLOT_INPUTS`. Existing process preparations are reused only when their
original source hashes, preparation definitions and output hashes still match.
There is no separate BDT plotter config or preparation script. Ordinary
campaign plots use the same main config without a study inventory.

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

## First all-sample look, October 12

The recovered 503-group campaign and seven additional signal points are
complete. Their combined frozen snapshot contains 187,394 events and 20,212
retained dimuon pairs. The 19 SM-bin histograms and nine signal points were
prepared into 12 process/point inputs under workspace
`validation/bdt_first_look_20261012/process_shapes/`. `bin_histogram_inputs.json`
pins both source manifests and the unchanged model. The SM snapshot retains
the original full-GEN normalization and explicitly records its 149 omitted
detector jobs; this is not a claim of full detector-campaign coverage.

`plots_sm/` and `plots_all/` contain the dedicated TEA overlays and coordinate
maps. `label_separation_annotated/` additionally contains direct genuine,
accidental and unknown score overlays for each process/point and combined SM,
with raw counts. Each label is normalized separately. Empty or nonpositive
signed-weight sums are reported as unavailable; negative bins are retained.
The display clone includes an exact score of one in its final bin, preserving
the original histogram and its overflow. These spectra retain training events;
held-out classifier performance is evaluated separately.

The combined-SM genuine spectrum has a material statistical limitation:
one genuine QCD pair supplies 96.5% of its weighted sum. This is annotated on
the combined panel. The nine signal points contain only 39 retained pairs,
so their first-look shapes do not establish mass/lifetime uniformity.
`plot_validation_annotated.json` links the original rendering audit and the
annotated figure receipt; the all-sample legend and direct separation panels
were visually checked. Twelve preparer tests and a native ROOT signed-weight,
zero-sum and endpoint-score overlay regression passed. Original score ROOT
files, histograms and earlier plots were preserved.
