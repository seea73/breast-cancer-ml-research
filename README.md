# Explainable Breast Cancer Analysis

An educational research project combining data analysis, mathematics,
and machine learning across three tasks: cell-measurement
classification, breast tissue image classification, and mammogram
region localization.

An interactive Streamlit app presents predictions, visualizations,
held-out evaluation results, and model limitations for each module.

## Module 1: Cell-Measurement Classification

Classifies breast-mass samples as benign or malignant using
30 numerical cell-nucleus measurements.

Dataset: Wisconsin Diagnostic Breast Cancer, UCI Machine Learning Repository.
https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic

## Methods

- Data-quality checks and exploratory analysis
- Stratified training/test split
- Five-fold cross-validation
- Logistic regression, SVM, and random forest comparison
- Decision-threshold analysis using out-of-fold training predictions
- Probability calibration assessment
- Global coefficients and individual prediction explanations
- Held-out evaluation with confidence intervals

## Results

Logistic regression was selected for the threshold experiment.

At the fixed threshold of 0.30, held-out test results were:
- Accuracy: 98.2%
- Sensitivity: 97.6% — 41 of 42 malignant samples
- Specificity: 98.6% — 71 of 72 benign samples
- ROC-AUC: 0.996
- One missed malignant sample and one false alarm

95% Wilson confidence intervals:
- Sensitivity: approximately 87.7%–99.6%
- Specificity: approximately 92.5%–99.8%

## Dashboard

- Select a dataset sample and view its prediction
- Inspect feature contributions
- Explore validation threshold tradeoffs
- Review fixed held-out test results

The sample selector includes training and test samples.
It demonstrates predictions rather than providing a new evaluation.

## Setup

Developed with Python 3.13.1.

Create a virtual environment:
```bat
python -m venv .venv
```

Activate it in Windows Command Prompt:
```bat
.venv\Scripts\activate.bat
```

Install dependencies:
```bat
python -m pip install -r requirements.txt
```

Run the dashboard:
```bat
python -m streamlit run app.py
```

## Files

- data/wdbc.data — source measurements and diagnosis labels
- data/wdbc.names — dataset documentation
- notebooks/01_cell_feature_classification_publication.ipynb — analysis
- results/ — saved model and result tables
- app.py — Streamlit dashboard
- requirements.txt — package versions

## Limitations

This project is not for clinical diagnosis.

Results come from one small held-out split of the source dataset.
External validation is needed. Feature explanations describe model
calculations and do not establish biological causes.

The model and thresholds were not changed based on test results.

## Module 2: Breast Tissue Image Classification

Classifies microscopic breast tissue images as benign or malignant
using the BreaKHis dataset.

Evaluation used 1,486 images from 13 held-out patient groups.

Held-out results at the fixed threshold of 0.1520:
- ROC-AUC: 0.659
- Balanced accuracy: 57.1%
- Malignant recall: 82.9%
- Benign recall: 31.4%

The Streamlit page supports image uploads and displays the model
score, threshold, prediction, and held-out performance.

Scores are not calibrated cancer probabilities. Low benign recall
means many benign images are classified as malignant.

Page: pages/02_Tissue_Image_Classification.py

## Module 3: Mammogram Region Localization

This module uses U-Net convolutional neural networks to highlight approximate abnormality regions in mini-MIAS mammograms.

The module is designed for educational research and comparison of image segmentation methods, not clinical diagnosis.

### Dataset

Dataset: mini-MIAS Mammography Database.

- 322 mammograms from 161 patients
- 207 normal and 115 abnormal images
- 119 lesion annotations with coordinates
- Four abnormal images excluded from localization because their location annotations were missing

Patient-group splitting prevents mammograms from the same patient appearing in different dataset splits.

| Dataset split | Patients | Eligible images |
|---|---:|---:|
| Training | 112 | 220 |
| Validation | 24 | 48 |
| Test | 25 | 50 |

Images are resized to 256 × 256 grayscale pixels and normalized to [0, 1].

Ground-truth masks are generated from the approximate circular annotations provided by mini-MIAS. Multiple annotated lesions are combined into a single mask.

These masks represent approximate localization regions, not precise lesion boundaries.

### Model Development

Three U-Net experiments were evaluated:

**U-Net v1 — Baseline**

- Three-level encoder-decoder architecture
- Binary cross-entropy plus Dice loss
- Original baseline model preserved

**U-Net v2 — Data Augmentation**

- Same core architecture as v1
- Added training augmentation
- Did not improve overall validation performance

**U-Net v3 — BCE + Tversky Loss**

- Same core U-Net architecture
- Modified loss to penalize false-positive predictions more heavily
- Approximately 487,000 trainable parameters
- Selected provisionally for Streamlit integration based on validation performance

All three experiments used the same patient-group dataset split.

### Validation Results

Models were compared using a common decision threshold of 0.15.

| Metric | U-Net v1 | U-Net v2 | U-Net v3 |
|---|---:|---:|---:|
| Mean abnormal Dice | 0.2511 | 0.2172 | 0.2582 |
| Mean abnormal IoU | 0.1806 | 0.1530 | 0.1847 |
| Abnormal images with any annotation overlap | 8/16 | 9/16 | 10/16 |
| Abnormal images with zero overlap | 8/16 | 7/16 | 6/16 |
| Normal images with highlights | 21/32 | 22/32 | 20/32 |
| Total false-positive pixels on normal images | 9,878 | 16,221 | 8,478 |

U-Net v3 achieved modest improvements over v1 in mean Dice, IoU, and false-positive pixel area on the validation set.

However, 6 of 16 abnormal validation images had no annotation overlap, and 20 of 32 normal validation images contained false highlights.

The validation set is small, and these results should not be interpreted as evidence of clinical effectiveness.

### Original Baseline Held-Out Test Results

The original U-Net v1 baseline was evaluated on a held-out test set using its earlier selected threshold of 0.35.

- Mean abnormal Dice: 0.2233
- Mean abnormal IoU: 0.1474
- Abnormal images with zero annotation overlap: 8/19
- Normal images with any highlight: 27/31
- Mean highlighted fraction on normal images: 0.69%

These are historical baseline results and are not directly equivalent to the common-threshold validation comparison above.

The test set was previously examined. Subsequent model development therefore relied on training and validation data.

No new independent held-out test performance is claimed for U-Net v3.

### Streamlit Application

The Mammogram Localization page supports:

- Uploading mammograms in supported image formats
- Selecting U-Net v1 (Baseline) or U-Net v3 (Experimental)
- Viewing the resized mammogram and predicted region overlay
- Displaying the highlighted image area
- Reviewing model-specific validation performance and limitations

Both models use a decision threshold of 0.15 in the current application.

Model outputs are not calibrated cancer probabilities. Highlighted regions do not establish malignancy, and absence of highlights does not exclude disease.

### Model Files

- `module3_mammogram/models/mammogram_unet_best.keras`
- `module3_mammogram/models/mammogram_unet_v3_best.keras`
- `pages/03_Mammogram_Localization.py`
- `notebooks/03_mammogram_localization_publication.ipynb`

The original baseline and experimental results are also preserved in the Google Drive research workspace.

### Dataset Reference

J. Suckling et al. (1994), "The Mammographic Image Analysis Society Digital Mammogram Database," Excerpta Medica, International Congress Series 1069, pp. 375–378.

Dataset source:
https://www.kaggle.com/datasets/kmader/mias-mammography

The downloaded mini-MIAS dataset is subject to its research-use licensing and attribution requirements.

### Limitations

- Small dataset with limited patient diversity
- Approximate circular annotations rather than precise lesion segmentation
- Low mean localization overlap
- Frequent false-positive highlights
- Some annotated abnormalities completely missed
- No independent external validation
- No established clinical reliability

This module is strictly an educational machine-learning research demonstration.

## Interpretation Across Modules

The three modules use separate datasets, models, and evaluation
methods. Their outputs are not combined into a patient diagnosis.

Module 3 learns approximate annotation circles, not precise lesion
boundaries. Its test results show limited overlap, missed annotated
regions, and frequent false highlights.

All modules are educational research demonstrations.

<!-- BEGIN STEP110 DOCUMENTATION -->

## Dataset Access, Licensing, and Attribution

The datasets used in this educational research project are
not included in this GitHub repository. Obtain each dataset
from its source and review the applicable terms before use.

### Module 1: Wisconsin Diagnostic Breast Cancer

- Source: UCI Machine Learning Repository
- Dataset: Breast Cancer Wisconsin (Diagnostic)
- URL: https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
- Reported license: Creative Commons Attribution 4.0 (CC BY 4.0)
- Citation: Wolberg, W., Mangasarian, O., Street, N.,
  and Street, W. (1993), Breast Cancer Wisconsin (Diagnostic),
  UCI Machine Learning Repository.
  https://doi.org/10.24432/C5DW2B

### Module 2: BreaKHis

- Source: Universidade Federal do Parana, Vision, Robotics
  and Imaging Laboratory.
- URL: https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/
- The source page describes non-commercial research use
  with acknowledgment of the source and citation of the
  original publication. It also displays CC BY 4.0
  attribution information. Users should review the
  applicable current terms before reuse or redistribution.
- Citation: Spanhol, F., Oliveira, L. S., Petitjean, C.,
  and Heutte, L. (2016), A Dataset for Breast Cancer
  Histopathological Image Classification, IEEE Transactions
  on Biomedical Engineering, 63(7), 1455-1462.

### Module 3: mini-MIAS

- Dataset: mini-MIAS mammographic image database.
- Information: http://peipa.essex.ac.uk/info/mias.html
- Archive information:
  https://www.andrew.cmu.edu/course/15-440-m14/applications/labs/lab4/old/mias.html
- The mini-MIAS materials describe scientific research use
  subject to their license agreement.
- Do not assume that the licensing of the separate,
  higher-resolution original MIAS archive automatically
  applies to mini-MIAS.
- Citation: Suckling, J., et al. (1994),
  The Mammographic Image Analysis Society Digital Mammogram
  Database, International Congress Series 1069, 375-378.

No original mammograms, histopathology images, or downloaded
dataset archives are distributed with this repository.

## Reproducibility and Model Availability

The repository publishes application code, research notebooks,
configuration files, aggregate evaluation metrics, and selected
research figures.

The trained model artifacts are intentionally excluded:

- `results/cell_classifier.joblib`
- `module_2/models/breast_tissue_classifier.keras`
- `module3_mammogram/models/mammogram_unet_best.keras`
- `module3_mammogram/models/mammogram_unet_v3_best.keras`

These files exist in the author's private research environment,
but they are not provided by this repository.

Consequently, cloning this repository and installing the Python
requirements is not sufficient to run all Streamlit prediction
features immediately.

Running those features requires the corresponding trained model
artifacts, supplied separately from an authorized source or
recreated through an appropriate training workflow.

Reproducing the reported results also requires the relevant
datasets, preprocessing procedures, patient-group partitions,
model initialization or checkpoints, and compatible software
dependencies.

The publication notebooks have their saved execution outputs
removed. Their inclusion does not establish that they can run
unchanged in a fresh environment.

Some experiments were developed in Google Colab using mounted
Google Drive directories. Researchers must review environment
paths and available files before attempting execution.

The pinned `requirements.txt` documents the project's package
versions, but installation and reproducibility on other
operating systems or Python versions have not been independently
verified.

This repository is an educational machine-learning research
portfolio, not a validated clinical diagnostic application.

<!-- END STEP110 DOCUMENTATION -->
