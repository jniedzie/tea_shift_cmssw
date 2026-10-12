# TEA SHIFT analysis instructions

The workspace `../AGENTS.md` applies here. Work directly on `main`; do not
create task branches or Git worktrees.

## Extend the main SHIFT configs

- When a histogram or plot fits the existing workflow, extend
  `configs/shift_histogrammer_config.py` and `configs/shift_plotter_config.py`.
  Do not create a small custom script, histogrammer, plotter or config for
  each routine study or feature.
- Extend the existing SHIFT histogrammer and its reusable analysis code for
  new calculations. Keep histogram definitions, study options and plot
  definitions in the two main configs.
- Reuse standard input, submission, ntuple-merging, histogram-merging and
  plotting tools. Extend existing shared helpers for provenance and full GEN
  normalization instead of introducing feature-specific config modules.
- Specialized diagnostics such as BDT ROC curves, fit checks, closure studies
  and uncertainty plots belong in scripts under `utils/` when they do not fit
  ordinary histogram/plot definitions. Keep these outside the main configs.
- Use a focused separate plot config when a study needs different inputs,
  normalization or presentation and supporting it would complicate the main
  config. Share reusable helpers; avoid a new config for each routine plot.
- Keep files configs short: paths and options belong there; inventory parsing
  and validation belong in existing shared input helpers.
- Name helpers after their actual purpose, such as preparing plot inputs.
- Follow the existing TEA config convention: input paths and shared options
  come from `shift_paths.py`. Do not add environment-variable switches for
  analysis paths or study selection when ordinary config settings suffice.
- Keep substantial algorithms and preparation code in reusable library code.
  The main configs should primarily define settings, histograms and plots;
  do not bloat them with calculations that do not belong there.
- Temporary investigation code belongs in a dated task directory outside the
  committed analysis sources. Preserve frozen production deployments and
  reproducible evidence; check dependencies before removing superseded files.
- Refactoring must preserve selections, event identities, native and sampling
  weights, full GEN denominators, uncertainty accounting and validation gates.
