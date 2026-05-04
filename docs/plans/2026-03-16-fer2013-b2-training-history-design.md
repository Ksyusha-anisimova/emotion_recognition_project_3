# Design: FER2013 B2 training history + plots

## Goal
Save training history and PNG plots when running `model/train_fer2013_b2.py` (3 classes on `data/fer2013`) into a dedicated subfolder for report screenshots.

## Scope
- Track per-epoch: train loss, val loss, train accuracy, val accuracy.
- Save `training_history.json` and `training_history.png` after training ends.
- Keep checkpoint saving behavior unchanged.

## Storage
- Default output directory: `model/checkpoints/fer2013_b2/`.
- History files are saved to the same directory as the checkpoint output file.

## Implementation plan
- Add history lists in `train_fer2013_b2.py`.
- Append metrics each epoch.
- Add helper functions:
  - `save_training_history(out_dir, history)`
  - `plot_training_history(out_dir, history)`
- Use matplotlib for PNG plot (loss + accuracy).

## Notes
- If `--output` is provided, history is saved to that file’s directory.
- If matplotlib is missing, training still runs but plots will fail; current environment already uses matplotlib in `model/train.py`.

## Git
No commit performed because `.git` is not present in the workspace.
