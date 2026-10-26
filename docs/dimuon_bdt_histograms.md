<!-- Code style: human-code. -->
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

The optional `shift_bdt_mixing_plotter_config.py` reads this inventory. Its shared
`utils/shift_bdt_mixing_histogram_inputs.py` helper applies
`cross_section_pb/full_gen_sumw` to each weighted SM bin before
combining bins into QCD, DY and J/psi histograms. It preserves raw counts and
keeps signal points separate. It writes `plot_inputs.json` only as complete
after every process histogram has been written and checksummed. It never uses
selected Nano event counts or selected weight sums as the GEN denominator.

Set the study paths in `configs/shift_paths.py`:

```python
bdt_mixing_histogram_inventory = "/absolute/path/bdt_bin_inputs.json"
bdt_mixing_preparation_directory = "/absolute/path/new_process_shapes"
bdt_mixing_plot_output = "/absolute/path/new_plots"
bdt_mixing_plot_scope = "all"
```

Then prepare the process mixture and render it in the ordinary TEA plotter:

```bash
source tea/setup.sh
cd bin
python plotter.py --config shift_bdt_mixing_plotter_config.py
```

`bdt_mixing_plot_output` selects the plot directory. The config
produces PDF score shapes and reconstructed-coordinate maps for the
inclusive and Both-Both categories. The three SM process distributions are
shown as lines and normalized separately to unit area after the bin mixture
has been corrected. MC label subsets have separate overlays. These plots
diagnose classifier separation and sculpting; efficiencies and AUC belong to
the performance report. `N_eff` refers to the MC normalization fitted to data,
and is not a classifier metric. The coordinate maps are not proper-lifetime
maps; a lifetime study requires the corresponding signal MC and truth labels
for evaluation.

New plot exports are PDF only, as requested after the October 12 study.
Project-accessible copies are in `plots/vertex_classifier/BDT_20261012/`; its index links directly
to PDFs. Original hash-locked validation archives retain their historical
renderings for provenance.

Set `bdt_mixing_plot_scope = "sm"` for the three SM process overlays, or `"all"`
(the default) to include each signal point. Use distinct
`bdt_mixing_plot_output` directories when rendering both. The all-sample view
uses an external legend when there are more than five curves and distinct
signal colors/styles. SM process legends are QCD, DY and J/psi after combining
their bins. The input helper resolves either equivalent EOS user mount while
retaining the original input path and checksum in its receipt.

The completed process manifest can also be supplied directly through
`bdt_mixing_histogram_inventory`. The inventory schema selects BDT or mixing
plots. Existing process preparations are reused only when their
original source hashes, preparation definitions and output hashes still match.
There is no separate preparation command. Ordinary campaign plots continue
to use `shift_plotter_config.py`; the optional study config handles BDT and
event-mixing overlays without loading canonical campaign samples.

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

## Fixed-model acceptance and label audit, October 12

Specialized classifier diagnostics use `utils/shift_bdt_acceptance.py` and
`utils/shift_bdt_label_audit.py`. They leave ordinary histogram definitions in
the main config and perform no fitting or ntuple selection. Acceptance
extraction checks exact parity with the completed first-look evaluation,
including integer physical-event identities, scores, labels, nuisance values
and weights. Truth positions and match diagnostics are separate from model
predictors. The comparison uses the existing validation thresholds for the
quality-only and uniformity BDTs, event-grouped uncertainties, signed weights,
explicit flow bins, topology and joint mass/z/radius cells. Shared paired
statistics are in `utils/shift_bdt_acceptance_statistics.py`.

Completed evidence is in workspace
`validation/bdt_acceptance_study_20261012/comparison/`: `results/comparison.json`,
CSV summaries, unchanged comparison scores, and `plots_reviewed/`. The source
snapshot, population/partition closure and output hashes are checked in
`validation.json`. The current uniformity model retains 88.83% genuine pairs
and rejects 49.85% accidentals, versus 86.65% and 53.04% for quality-only.
The simpler model does not remove DY mass dependence and has stronger radius
dependence. Weighted accidental support remains inadequate; neither cut is
a validated physics selection.

`label_audit/` independently traces the dominant QCD pair and normalization,
audits unknown matches and label conventions, and distinguishes common origin
from fitted-vertex precision. Unknowns comprise 18.32% of all retained pairs
and increase with radius and poorer detector coverage. Angular truth links
are diagnostics and never replace hit-based labels. The separate
`comparison/label_identification_limits.json` gives possible unknown-label
assignment bounds; these are not physics uncertainties or corrections.

Fourteen focused comparison/statistics tests and seven label-audit tests pass.
`comparison/paired_support.json` separately annotates support for all 180
existing paired differences. A mathematically available weighted ratio or
bootstrap interval does not imply adequate independent-event support.
The reused identification-bound tests cover 1,555 exhaustive patterns. All
original first-look reports and artifacts are preserved. Scientific decisions
and remaining acceptance gates are recorded in the classifier task log;
reconstruction findings remain in `SHIFT_RECONSTRUCTION.md`.

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

## October 13 DNN and vertex-category study

The study compares frozen BDT references with validation-selected neural
models. Specialized score, ROC, category and signal-transfer diagnostics use
`utils/shift_vertex_dnn_diagnostics.py`,
`utils/shift_bdt_category_diagnostics.py` and
`utils/shift_bdt_signal_transfer.py`. Ordinary ntuple histograms and their
deployed BDT are unchanged.

Final PDF figures, numerical results, models and provenance are under
[`plots/vertex_classifier/DNN_vertex_20261013/`](../plots/vertex_classifier/DNN_vertex_20261013/README.md), inside
this repository for direct access in VS Code. The reviewed six-model figures
are in `dnn_comparison_reviewed/`, category figures in `dnn_categories/`, and
the independent six-point signal comparison in
`fresh_signal_transfer/evaluation_v2/`. Earlier draft renders are preserved.
Generated study artifacts remain local and are ignored by Git.

The category-aware DNN keeps 88.8% of genuine Near-Both SM pairs versus
78.4% for the flatter BDT, with essentially the same pooled fake rejection
(41.0% versus 40.6%). It keeps 160/173 genuine independent signal pairs
versus 147/173 for that BDT; the strong BDT keeps 173/173. This is improved
preservation, not demonstrated better pooled rejection or mass/lifetime
independence. QCD has no genuine test support. Models and cuts are locked
before neural test and fresh-signal inference; unknown labels stay separate,
and all 1,200 fresh GEN parents remain in the denominator audit.

The frozen source and detailed physics record are in
`CMSSW_17_0_0_pre4/src/PhysicsTools/ShiftDimuonClassifier/` and its
`DIMUON_VERTEX_CLASSIFIER.md`. Validation passed 132 classifier tests and
18 targeted analysis tests. Future final results and PDF figures must remain
inside this repository.

The subsequent one-hour information study adds 32 fits and 69 reconstructed
inputs (associated muon quality/hits/timing and five event occupancy counts).
Results are in
[`plots/vertex_classifier/DNN_vertex_information_20261013/`](../plots/vertex_classifier/DNN_vertex_information_20261013/README.md).
Both rich pooled-model seeds pass the reconstructed development acceptance
checks, rejecting 49.10/49.27% of fakes versus 40.95% for the earlier
constrained DNN. They keep 90.80/89.93% of genuine SM pairs and 163/173 or
152/173 genuine signal pairs. Signal seed variation remains unresolved.
No generated quantity enters inference, no existing labels/cuts are changed,
and this is an information ablation on the same training events. Reviewed
PDFs are in `dnn_comparison_reviewed/`, `categories_v2/` and
`signal_transfer/` within that study. The original study commits are CMSSW
`0ded56efdf9` and TEA `a890545`; the new study is not committed.

The next response study separates classifier retention from the full generated
event response. Its final PDF figures and numerical evidence are in
[`plots/vertex_classifier/Vertex_efficiency_20261013/`](../plots/vertex_classifier/Vertex_efficiency_20261013/README.md).
The six complete points retain all 1,200 GEN parents and use proper lifetimes
from timed GEN audits. Incomplete points remain unknown. The rich DNN retains
163/173 or 152/173 genuine signal events, corresponding to 13.6% or 12.7% of
all generated events; the stronger BDT retains 173/173, or 14.4% of all GEN.
Local lifetime reweighting is implemented but lacks the required effective
selected-event support. This is simulated reconstruction and selection response,
with real recording acceptance and J/psi data corrections still unresolved.

`utils/shift_vertex_signal_cache.py`, `utils/shift_vertex_signal_scores.py` and
`utils/shift_vertex_efficiency_map.py` process explicitly qualified complete
signal points with the existing frozen inputs, models and cuts. They preserve
event identities, native weights and zero-candidate parents, and refuse
overwrite. New stage-2 simulation is already owned by DAG 12911694; do not
submit competing signal production or change its sealed deployment.

## October 13 complete-input BDT/DNN and training-size comparison

The next comparison uses all 119 reconstructed variables and five explicit
category columns for every model. It includes the single-muon and event
variables. Each of two separate domains has 81 completed fits: three BDT
settings and three DNN architectures, classification or decorrelation loss,
25/50/100% nested event-grouped training samples, and three seeds. The current
sample uses all 187,214 compatible completed SM events. The newly recovered
historical binned production contains 329,997 events, but uses the old
IR1/ATLAS proxy geometry and different generation/decay settings. It is a
separate stress test; its very strong process/class imbalance makes pooled
performance insufficient evidence of vertex discrimination.

Current development performance at full training size is 77.98% fake
rejection with 89.86% genuine retention for the regularized depth-five BDT,
versus 73.41--73.57% rejection with 90.39--91.33% retention for the shared
64/32 DNN. Both improve with increasing training population. Neither current
model family passes the fixed acceptance-uniformity requirements. These are
reused development events, not independent confirmation.

Final PDF ROC, score, mass/position/category and learning-curve diagnostics
are in
[`plots/vertex_classifier/Vertex_complete_20261013/`](../plots/vertex_classifier/Vertex_complete_20261013/README.md),
with frozen sources, numerical evidence and independent hash/selection audits.
Six Condor jobs, `12913514.0` through `.5`, were submitted for a separate
81-fit current-domain comparison of moderate decorrelation and direct soft
acceptance losses. Its protocol was frozen before the initial development
evaluation. Read the study README and `current_refinement/condor/submission.json`
before resuming. Wait for all archived fit receipts and inspect the six jobs
before locking families and generating its development or signal figures.
No new classifier or selection cut is deployed in the ntuples or histograms.

The October14 overall recommendation is the two-member rich119-input DNN
ensemble at mean score>=0.3016294240951538. It retains90.27% of genuine SM
development pairs and rejects52.04% of fakes;30 new signal points retain
701/760 known genuine pairs. The selected score retains reconstructed
category information through its existing topology inputs. See the single
recommended model and final PDFs in
[`plots/vertex_classifier/Vertex_best_20261014/`](../plots/vertex_classifier/Vertex_best_20261014/README.md).
Selection uses validation rejection, dependence and member-omission stability;
the signal grid never changes the cut. Reconstructed proper length is a
nominal-target proxy, with true mean proper lifetime shown separately from
GEN audits. Finer bins still expose reconstructed tails and many signal
points have insufficient support. This is a working choice, not a final
physics cut or validated J/psi luminosity correction. Existing production
scores and ordinary histogram selections are unchanged.
