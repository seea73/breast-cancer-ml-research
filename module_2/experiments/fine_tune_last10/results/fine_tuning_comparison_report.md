# MobileNetV2 Fine-Tuning Experiment Report

## 1. Objective

Evaluate whether reducing the number of trainable MobileNetV2
backbone layers from the last 30 to the last 10 improves validation
performance for BreaKHis breast histopathology image classification.

## 2. Dataset

- Dataset: BreaKHis
- Total images: 7,909
- Unique extracted patient IDs: 81
- Training: 5,303 images, 56 patient groups
- Validation: 1,120 images, 12 patient groups
- Held-out test: 1,486 images, 13 patient groups
- Classes: Benign and malignant
- Input size: 224 x 224 RGB
- Patient-group overlap: None detected

## 3. Experimental Method

Both configurations use MobileNetV2 and the original Stage 1
checkpoint as the starting point.

The experiment fine-tunes the last 10 backbone layers instead of
the original last 30, with BatchNormalization layers frozen.

- Learning rate: 0.00001
- Maximum Stage 2 epochs: 20
- Model selection: Validation ROC-AUC
- Early stopping patience: 5
- Class weighting: Applied during training

The original Stage 1 model was reused without retraining.

## 4. Training Results

| Metric | Last 30 layers | Last 10 layers |
|---|---:|---:|
| Stage 2 epochs | 20 | 7 |
| Best epoch | 16 | 2 |
| Best validation ROC-AUC | 0.7577 | 0.6773 |

Difference in best validation ROC-AUC (last 10 minus last 30):
-0.0804

## 5. Experimental Model Validation

Results from the saved last-10-layer checkpoint:

- Validation images: 1120
- ROC-AUC: 0.6774
- Average precision: 0.8583
- Validation-selected threshold: 0.188268
- Accuracy: 80.18%
- Balanced accuracy: 66.35%
- Malignant recall: 92.00%
- Benign recall: 40.70%
- Precision: 83.83%
- F1-score: 87.72%

The classification threshold was selected using validation data
to maximize image-level balanced accuracy.

## 6. Interpretation

The last-30-layer configuration achieved a higher best validation
ROC-AUC than the last-10-layer experiment.

The last-10-layer model achieved relatively high malignant recall
at its selected threshold, but benign recall remained low, indicating
many benign images were classified as malignant.

The results support retaining the original last-30-layer model as
the stronger candidate on this validation split.

## 7. Limitations

- The comparison uses a single patient-group split.
- Validation data was used for model and threshold selection.
- The experiments used different numbers of completed epochs.
- Training randomness may affect the comparison.
- The experimental model was not evaluated on the test set.
- These image-level results do not establish clinical performance.
- The model is intended for educational research only.

## 8. Conclusion

Fine-tuning the last 10 MobileNetV2 backbone layers did not improve
validation ROC-AUC compared with the original last-30-layer approach.

The comparison is exploratory and does not establish statistical
significance or generalization to independent clinical data.
