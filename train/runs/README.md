# Run artifacts (checkpoints, metrics) land here. Contents are gitignored.

Logging: code never uses bare `print` — it uses `log()` from
`train/utils/log.py` (drop-in replacement; console output is opt-in via
`print_console=True`). Quick scripts write `common.log` in the repo root by
default; training runs should pass `filename=` pointing into their run
directory here. Inspect any run with `tail` on its log file.
