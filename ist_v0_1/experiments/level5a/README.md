# Level 5A ablation

`results.json` is the two-seed pilot run for the shared-data ablation framework.
The shortened 600/100/100-step schedule did not teach any variant the remote
query task, so these pilot accuracies must not be interpreted as an architecture
ranking. They validate the runner, metrics, parameter accounting and output
format only.

The confirmatory run must use enough training for the first 128-token stage to
reach at least 90% validation accuracy before advancing. Compare final accuracy,
steps-to-threshold and accuracy AUC only after that gate passes.
