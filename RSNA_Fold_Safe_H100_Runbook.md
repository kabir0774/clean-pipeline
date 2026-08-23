# RSNA: fold-safe weak labels and H100 OOF training

## Completed artifacts

- `build_rsna_fold_safe_labels.py`: builds five label files from V2 + GPT sources. Each fold’s target-specific V2/GPT mixture is selected only from that fold’s training gold records. The held-out gold studies are disabled from training.
- `fold_safe_labels/`: generated successfully from 4,407 studies and 58 gold rows.
- `train_rsna_soft_oof.py`: trains an MRI-only 2.5D model against soft weak labels, evaluates only on each held-out gold fold, and saves full gold OOF predictions.

## Removed loopholes

- No compressed calibration payload.
- No use of test-series metadata as a fitted regressor.
- No external competition head/checkpoint or inherited ensemble branch.
- No report text as a model input.
- No globally selected V4 blend inside fold validation.
- No public-leaderboard-selected blend or architecture weights.

## First H100 experiment

Copy the three `.py` scripts plus `fold_safe_labels/` to the H100 machine. First run only `resnet18` to establish a reliable baseline.

```bash
python train_rsna_soft_oof.py \
  --root /path/to/competition-data \
  --labels-dir /path/to/fold_safe_labels \
  --output runs/b0_resnet18 \
  --architecture resnet18 \
  --epochs 12 --batch-size 3 --workers 6 --compile
```

Required result:

- `runs/b0_resnet18/resnet18_gold_oof.csv`
- `runs/b0_resnet18/resnet18_metrics.json`

Use only `macro_gold_oof_auc` and the 12 per-target AUCs to select the next experiment.

## Architecture plan

1. `resnet18`: baseline and data-pipeline verification.
2. `convnext_tiny`: first capacity/diversity test if ResNet baseline works.
3. `efficientnet_v2_s`: third architecture only if ConvNeXt changes errors or improves repeated OOF AUC.

Run architectures one at a time. Do not train all three and then pick based on public leaderboard.

```bash
# after the baseline is complete
python train_rsna_soft_oof.py --root /path/to/competition-data --labels-dir /path/to/fold_safe_labels --output runs/b1_convnext --architecture convnext_tiny --epochs 12 --batch-size 2 --workers 6 --compile
```

## Go/no-go requirements

Promote a configuration only if:

- it improves macro gold OOF AUC by at least 0.002–0.003;
- the gain appears in at least three of five gold folds;
- no target drops by >0.005 without an explained trade-off;
- a second seed reproduces the gain before use in an ensemble.

The model has not been run in this chat: an H100 training environment is required. Do not call a public score or a weak-label score the model’s final AUC.
