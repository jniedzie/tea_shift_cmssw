<!-- Code style: human-code. -->
# TEA SHIFT analysis instructions

The workspace `../AGENTS.md` applies here. Work directly on `main`; do not
create task branches or Git worktrees.

## Code readability labels

Each maintained source, configuration, test, utility and local documentation
file starts with a `Code style: human-code` or `Code style: AI-code` comment.
Read that label before editing it and preserve it unless the user changes it.

- **human-code** files are read line by line by the user. Make small,
  transparent changes with direct control flow, ordinary names and local
  comments that explain the reason for a non-obvious step. Do not introduce
  framework-like abstractions, dynamic path discovery, dense comprehensions or
  large refactors solely to make the code shorter.
- **AI-code** files may contain substantial generated machinery, such as
  merging or validation utilities. New AI-code files must begin with the label
  and say that they were AI-generated. Keep their public interface, errors and
  evidence clear enough to audit.
- The user may relabel a file. On the next edit, follow the new label; when a
  file becomes human-code, simplify the affected area as part of the work when
  practical.

Treat existing analysis-facing files as human-code unless they are already
marked AI-code. This includes `shift_paths`, `shift_params`, all ordinary
configs, file configs and `ShiftHistogramsFiller`.

## Extend the main SHIFT configs

## Final analysis results and plots

- Put every final, user-facing analysis result in this repository. Final
  figures belong in `plots/<study>/`, so they are directly visible from the
  project in VS Code. Do not leave a final figure only in an AFS validation,
  scratch, batch, or temporary-output directory.
- A job may write an intermediate figure elsewhere while it runs. Before
  reporting results, publish the final PDF here, verify its SHA-256 digest,
  and remove the external duplicate when it has no remaining provenance role.
  Keep receipts and non-plot numerical evidence with the study.
- Prefer PDF for final figures. Create PNG only when the user specifically
  asks for it.

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
