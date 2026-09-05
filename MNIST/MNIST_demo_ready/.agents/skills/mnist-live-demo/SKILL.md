---
name: mnist-live-demo
description: Use when implementing or fixing the MNIST CSV classification experiment in this repository.
---

# MNIST live demo

Use the supplied training file for fitting and the supplied test file for evaluation. Generate the metrics, predictions, figures, and report from the same prediction run.

Use small synthetic inputs for most tests, and load the full dataset only in the experiment command. Test for common CSV mistakes: treating the first row as a header, including the label among the pixels, fitting on test rows, and reshaping pixels in the wrong order.

Write `REPORT.md` from the generated artifacts. Check both figures and compare the finished files with `expected/README.md` before handing the project back.
