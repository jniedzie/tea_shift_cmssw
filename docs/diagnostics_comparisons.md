<!-- Code style: human-code. -->
# Comparing reconstruction versions

From the analysis directory, run:

```bash
python3 utils/shift_diagnostics_plotter.py --compare-versions v40 v41
```

Numeric versions (`40 41`) also work. Each version must resolve to exactly one
`plots/vN_<hash>[_<recoVariant>]/histograms.root`. Use `--histograms-dir` to
search elsewhere. `--output-dir` changes the output parent directory.
Comparison mode is exclusive with `--input` and `--version`.

Outputs go to `plots/comparisons/v40_vs_v41/`:

- Four scale/resolution PDFs: muon and dimuon, each split into constrained and
  unconstrained. Colors identify versions. Points show `1 + mean`; bars show
  RMS, not uncertainty on the mean, exactly as in the single-version summary.
- Two topology-fraction tables: unconstrained reference-histogram entries,
  including underflow/overflow, divided by the sum of topology entries.
  Errors are binomial. Dimuon Inclusive is excluded from that sum.
- Two inclusive efficiency PDFs versus generator coordinates, retaining the
  existing Clopper–Pearson intervals and dimuon logarithmic scale.
- Two all-category efficiency tables: integrated efficiency and passed/total
  counts, including underflow/overflow, using the pT histograms as in the
  original legend. Each category uses its own stored truth denominator.

Tables are saved as both PDF and Markdown. `inputs.json` records the exact
input paths, including reconstruction variants. Missing required histograms
or inconsistent efficiency pairs stop generation before output is written.
Empty topology samples or efficiency denominators are shown as N/A.

The dimuon efficiency canvas also includes the lab-frame opening angle between
the generated daughter momenta, in radians. Its logarithmic angle axis resolves
small separations while covering the full range up to pi. The numerator uses
the same truth angle as the denominator and the existing hit matching and
topology selections. Regenerate histograms to populate this panel; older files
still render the five existing variables and display a missing-angle notice.
If any comparison input lacks the angle histograms, the angle comparison is
marked unavailable for all versions. A partial or inconsistent pair still
fails the comparison checks.
