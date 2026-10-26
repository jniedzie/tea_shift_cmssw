<!-- Code style: human-code. -->
# Plot directories

Keep plots under one of these five topics. Use study or campaign directories
inside each topic; put supporting numerical evidence next to its figures.

| Directory | Contents |
| --- | --- |
| [`physics/`](../plots/physics/) | User-facing distributions, signal efficiencies and lifetime plots. |
| [`reconstruction/`](../plots/reconstruction/) | Reconstruction development, detector checks and version comparisons. |
| [`limits/`](../plots/limits/) | Expected limits, sensitivity studies and fit checks. |
| [`vertex_classifier/`](../plots/vertex_classifier/) | BDT/DNN studies, selected models and classifier validation. |
| [`production/`](../plots/production/) | Sampling, transport, event mixing and production checkpoints. |

The ordinary plotter writes SM results to
[`physics/SM/`](../plots/physics/SM/) and signal overlays to
[`physics/SM_BSM/<campaign>/`](../plots/physics/SM_BSM/).
Versioned reconstruction histograms and plots live in
[`reconstruction/versions/`](../plots/reconstruction/versions/), with comparisons
in its `comparisons/` subdirectory.
The selected classifier report is
[`vertex_classifier/Vertex_best_20261014/README.md`](../plots/vertex_classifier/Vertex_best_20261014/README.md).

The 26 October 2026 cleanup retained the detailed studies, models, source
snapshots and accounting records. It removed five disposable plotting-test
PDFs, an empty mixing-study directory and an already broken EOS shortcut.
Older studies are historical evidence, not current physics results.

Exact old-to-new paths and deletion hashes are recorded in
[`validation/plot_directory_cleanup_20261026/`](../../validation/plot_directory_cleanup_20261026/).
Frozen receipts and source snapshots retain their original path strings as
provenance; translate those paths using `path_mapping.json` when replaying an
old study. Do not recreate the old top-level layout.
