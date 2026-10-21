# SLM

The following commands exist:

* `smoke`: runs one training step with a small model config to test the environment and dependencies.
* `train`: runs a full training session with the specified configuration.
* `report`: generates a report of the training results, including evaluation metrics and visualizations.

All commands suppport the following options:
* `--experiment EXPERIMENT` YAML filename in experiments/, e.g. text8_weighted
* `--run-dir RUN_DIR`
* `--data-dir DATA_DIR`
* `--device DEVICE`
* `--wandb-mode {online,offline,disabled}` tracking mode (default: online)
* `--project PROJECT`     W&B project; reports append -report to this or the training project
* `--entity ENTITY`       W&B account or team
* `--name NAME`           W&B run name; defaults to the experiment filename stem

## Run an experiment
The folder `experiments/` contains different configuration yaml files.

Select one configuration and run `train sudoku` or `train text8`:

```bash
uv run --extra gpu slm train sudoku --experiment sudoku_weighted --run-dir runs/sudoku
```

Other flags:
* `--resume`              resume run-dir/last.pt
* `--max-hours` MAX_HOURS
* `--stop-after-updates` STOP_AFTER_UPDATES
* `--checkpoint-every` CHECKPOINT_EVERY
* `--checkpoint-seconds` CHECKPOINT_SECONDS

## Report
Once training is done, you can generate a report with:

```bash
uv run --extra gpu slm report sudoku --experiment sudoku_weighted --run-dir runs/sudoku

CUDA_VISIBLE_DEVICES=0 uv run --no-dev --no-sync slm report sudoku     --experiment sudoku_weighted_rating0   --run-dir runs/sudoku_weighted_rating0     --report sweep    --sampler budgeted    --sampling-steps 128  --sampling-seeds 

CUDA_VISIBLE_DEVICES=0 uv run --no-dev --no-sync slm report sudoku     --experiment sudoku_weighted_rating0   --run-dir runs/sudoku_weighted_rating0     --report default    --sampler budgeted    --sampling-steps 128  --sampling-seeds 
```

Current report types:
* `default`: basic report with metrics and plots
* `sweep`: will run over different steps and compare the results
* `heatmap`: one-board run for detailed analysis of the model's behavior on a single board

Currently, exist this sampling algorithms:
* `default`: the original sampler
* `protected`: protect the highest confidence candidate from being prune
* `confidence-gated`: prune cells only with high prob. + 'protected' sampler
* `topk`: select the topk most confident cells, hardcode retention probability to 0.5
* `budgeted`: based on topk, instead of taking the most confident cells, prune the ones who attribute the less. also respect the scheduler

Other flags:
* `--sampling-seed` SAMPLING_SEED Run once using this seed
* `--sampling-seeds` SAMPLING_SEEDS Generate N distinct random seeds and average N runs
* `--sampler` {default,protected,confidence-gated,topk,budgeted}
* `--report` {default,sweep,heatmap}
* `--legacy`              Load a legacy notebook Sudoku checkpoint
* `--board-id` BOARD_ID   Heatmap: zero-based index in the selected test split
* `--weights` {ema,raw}   Heatmap: checkpoint weights to inspect
* `--sampling-steps` SAMPLING_STEPS Default/heatmap sampling budget (defaults to report_steps or sample_steps)
