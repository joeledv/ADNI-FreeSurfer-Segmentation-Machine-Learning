# ADNI-FreeSurfer-Segmentation-Machine-Learning
## MRI Radiomics Processing Pipeline

This repository contains tools for processing MRI data and extracting radiomics features.

### Installation

#### Prerequisites
- Python 3.7+
- pip
- Git (only needed if direct installation fails)

#### 1. Install PyRadiomics

First, ensure your Python environment is up to date:

```bash
pip install --upgrade pip setuptools wheel
pip install pyradiomics==3.0.1
```

If installation fails, try building from source:
```bash
git clone https://github.com/AIM-Harvard/pyradiomics.git
cd pyradiomics
pip install .
```

Install aditional requirements:
```bash
pip install nibabel SimpleITK nilearn pandas numpy
```

## Patient Selection

For this project, the initial step involved requesting access to the Alzheimer's Disease Neuroimaging Initiative (ADNI) database. The focus was placed on patients who were initially diagnosed with Mild Cognitive Impairment (MCI), specifically identifying those who progressed to Alzheimer's Disease (AD) within a 24-month time frame. This allowed the construction of a dataset comprising two groups: MCI converters (patients who developed AD within 24 months) and MCI non-converters (patients who remained stable during the same period).

## Correlation Analysis

Within the ADNI dataset, there exists a pre-extracted feature set that was used as a reference to validate our own radiomic feature extraction pipeline using PyRadiomics. Both pipelines employed FreeSurfer for anatomical segmentation of brain regions. A comparative analysis was conducted on features such as area and volume across various brain regions. Regions that showed low consistency or poor discriminative performance during the analysis were excluded from further machine learning model development.

## Exploratory Data Analysis 

During the exploratory data analysis, our initial step was to remove columns with a standard deviation lower than 0.01, as these features provided little to no variability and were considered redundant. Additionally, we eliminated highly correlated feature pairs, retaining the one with the higher variance in each case. Despite these efforts, the dataset still presented a dimensionality issue, having more features than patient samples. To address this, we applied Recursive Feature Elimination with Cross-Validation (RFECV), using a Logistic Regression model and optimizing for the Area Under the Curve (AUC) metric on the validation set, in order to identify the most informative features for subsequent modeling.

### Dataset
https://www.kaggle.com/datasets/joeldv1/adni-change-mci-to-alzheimer-in-24-months

### References
- Alzheimer's Disease Neuroimaging Initiative. (n.d.). ADNI: Alzheimer's Disease Neuroimaging Initiative. Retrieved from https://adni.loni.usc.edu
- Danvalcor, jcg01, DairaRe. (2024). ADNI-FreeSurfer-Segmentation. Retrieved from: https://github.com/Danvalcor/ADNI-FreeSurfer-Segmentation
