# Debug and CMD Pack

Put these files in the project root.

Recommended files in root:

```text
camelyon16_audit_project_v2/
  run_pipeline.py
  run_pipeline_debug.py
  cmd_runner.py
  run_debug.cmd
  run_fake_demo.cmd
  run_audit_only.cmd
  run_viewer.cmd
  run_api.cmd
  install_requirements.cmd
  clean_outputs.cmd
  generate_fake_results.py
  generate_fake_results_presentation.py
  src/
  data/
  outputs/
```

## Best commands

Install packages:

```cmd
install_requirements.cmd
```

Generate demo results and open viewer:

```cmd
run_fake_demo.cmd
```

Run debug pipeline:

```cmd
run_debug.cmd
```

Run debug pipeline without training:

```cmd
run_debug_skip_train.cmd
```

Run audit only:

```cmd
run_audit_only.cmd
```

Open viewer only:

```cmd
run_viewer.cmd
```

Open API only:

```cmd
run_api.cmd
```

Python command style:

```cmd
python cmd_runner.py env
python cmd_runner.py fake
python cmd_runner.py debug
python cmd_runner.py debug-skip-train
python cmd_runner.py audit-only
python cmd_runner.py viewer
python cmd_runner.py api
```

The debug pipeline writes:

```text
outputs/logs/debug_environment.json
outputs/logs/debug_stage_timing.json
outputs/logs/postflight_report.json
outputs/logs/train_mil.log
```
