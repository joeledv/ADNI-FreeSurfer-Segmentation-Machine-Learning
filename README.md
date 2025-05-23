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
