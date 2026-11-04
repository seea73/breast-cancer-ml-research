# MobileNetV2 Last-10-Layer Fine-Tuning Experiment

## Overview

This experiment investigates whether fine-tuning fewer MobileNetV2
backbone layers improves breast histopathology image classification
on the BreaKHis dataset.

The original configuration fine-tuned the last 30 backbone layers.
This experiment fine-tuned the last 10 backbone layers.

Both configurations use the original trained Stage 1 checkpoint
as their starting point.

**Purpose:** Educational machine learning research.
**Not intended for clinical diagnosis or medical decision-making.**

## Dataset

- Dataset: BreaKHis
- Total images: 7,909
- Extracted patient identifiers: 81
- Training: 5,303 images from 56 patient groups
- Validation: 1,120 images from 12 patient groups
- Original test partition: 1,486 images from 13 patient groups
- Classes: Benign (0), Malignant (1)
- Input: 224 x 224 RGB images

The split was created using extracted patient identifiers.
No patient-group overlap was detected across partitions.

The original test partition was not evaluated in this experiment.

## Model Configuration

- Architecture: MobileNetV2
- Starting weights: Original best Stage 1 checkpoint
- Fine-tuning: Last 10 backbone layers
- Earlier backbone layers: Frozen
- BatchNormalization: Frozen
- Classification head: Trainable
- Recorded trainable parameters: 726,721
- Optimizer: Adam
- Initial fine-tuning learning rate: 0.00001
- Maximum Stage 2 epochs: 20
- Early stopping patience: 5
- Checkpoint selection: Best validation ROC-AUC
- Class weighting: Applied
- Random seed: 42

Input rescaling from [0, 255] to approximately [-1, 1]
is included in the saved model.

## Fine-Tuning Comparison

| Metric | Last 30 Layers | Last 10 Layers |
|---|---:|---:|
| Stage 2 epochs completed | 20 | 7 |
| Best epoch | 16 | 2 |
| Best validation ROC-AUC | 0.7577 | 0.6773 |

**Difference (last 10 minus last 30):**
-0.0804 ROC-AUC.

## Experimental Model Validation

The best last-10-layer checkpoint was reloaded and evaluated
on the validation partition.

- Validation images: 1120
- Validation ROC-AUC: 0.6774
- Average precision: 0.8583
- Selected classification threshold: 0.188268
- Accuracy: 80.18%
- Balanced accuracy: 66.35%
- Malignant recall: 92.00%
- Benign recall: 40.70%
- Precision: 83.83%
- F1-score: 87.72%

The threshold was selected on the validation set using
image-level balanced accuracy.

## Findings

The original last-30-layer model achieved higher validation
ROC-AUC than the experimental last-10-layer model.

Reducing the number of trainable backbone layers did not
improve validation discrimination in this experiment.

The experimental model achieved high malignant recall at
its validation-selected threshold, but benign recall remained low.

This indicates a substantial false-positive limitation.

## Limitations

- Only one patient-group split was investigated.
- Results are based on image-level validation metrics.
- Model selection and threshold selection used validation data.
- Different training durations and stochastic training effects
  limit causal interpretation of the comparison.
- No additional test-set evaluation was performed in this experiment.
- The broader project's original test set had previously been examined.
- No statistical significance or external validation was established.
- These results do not establish clinical reliability.

## Experiment Files

### Notebook
- `02_fine_tune_last10.ipynb`

### Models
- `models/mobilenetv2_stage2_best.keras`
- `models/breast_tissue_classifier.keras`
- `models/model_config.json`
- `models/threshold_config.json`

### Results
- `results/stage2_history.csv`
- `results/stage2_training_log.csv`
- `results/validation_predictions.csv`
- `results/validation_threshold_comparison.csv`
- `results/fine_tuning_comparison_report.md`

### Figures
- `figures/training_history.png`
- `figures/training_image_samples.png`

## Reproducibility Notes

The experiment requires the original trained Stage 1 checkpoint,
the BreaKHis dataset, and the project dependencies.

The notebook documents the patient-level split,
preprocessing, training configuration, and validation procedure.

To repeat training, use a separate experiment directory so that
existing checkpoints and result files are not overwritten.

Reproducing the exact numerical results may depend on
software versions and nondeterministic GPU operations.

The original BreaKHis images are not distributed with
this experiment's result files.

## Conclusion

The last-30-layer configuration remains the preferred
candidate based on validation ROC-AUC.

The last-10-layer configuration is retained as a documented,
reproducible experimental comparison.

Further evaluation on independent patient data would be
needed before making broader generalization claims.

<!-- BEGIN STEP110 EXPERIMENT GUIDANCE -->

## Environment and Reproducibility Guidance

The publication notebook for this experiment is:

`02_fine_tune_last10_publication_draft.ipynb`

The original experiment was developed in Google Colab with
project artifacts stored in Google Drive.

### Environment requirements

- Google Colab or another compatible TensorFlow environment.
- Access to BreaKHis under its applicable usage terms.
- The experiment's original patient-group training and
  validation partitions.
- The original trained MobileNetV2 Stage 1 checkpoint.
- Compatible Python and machine-learning dependencies.
- A separate writable directory for experimental outputs.

The publication notebook contains project-specific Google
Drive path references, including the original project layout
under `/content/drive/MyDrive/Breast_Cancer_Project`.

These paths must be reviewed and adjusted for a different
research environment.

### Important execution precautions

This notebook is provided to document the experimental
methodology and support review of the research workflow.

It is not a one-click, independently verified reproduction.

Do not execute all cells against the original research
workspace without inspecting their file paths and effects.

Some cells may train models or create and overwrite experiment
artifacts. Use a separate working copy of the project and
checkpoint files for any reproduction attempt.

The original model checkpoints, datasets, patient-level
prediction files, and private experiment backups are not
included in the public repository.

### Interpretation of results

The last-10-layer and original last-30-layer configurations
were compared using validation data.

The last-10-layer experimental model was not newly evaluated
on an independent held-out test set.

The results describe this particular dataset partition and
experimental setup. They do not establish clinical safety,
diagnostic reliability, or generalization to other populations.

<!-- END STEP110 EXPERIMENT GUIDANCE -->
