# Immunocompromised Mortality Prediction

This repository contains the code for the paper "Explainable machine learning model for prediction of 28-day all-cause mortality in immunocompromised patients in the intensive care unit: a retrospective cohort study based on mimic-iv database".

## Project Overview

This project develops and validates explainable machine learning models to predict 28-day all-cause mortality in immunocompromised patients admitted to the ICU. The analysis uses data from the MIMIC-IV database, encompassing ICU admissions at Beth Israel Deaconess Medical Center from 2008 to 2019.

## Code Structure

The repository is organized as follows:

```
immunocompromised-mortality-ml/
├── data_processing/
│   └── preprocessing.py
├── models/
│   ├── model_training.py
│   └── cox_analysis.py
├── evaluation/
│   ├── metrics.py
│   └── shap_analysis.py
├── visualization/
│   └── plotting.py
├── main.py
├── utils.py
└── requirements.txt
```

- `data_processing/`: Modules for loading and preprocessing the data
- `models/`: Modules for training machine learning models and Cox regression analysis
- `evaluation/`: Modules for evaluating model performance and interpretability
- `visualization/`: Modules for creating visualizations and plots
- `main.py`: Main entry point for running the complete workflow
- `utils.py`: Utility functions used throughout the project
- `requirements.txt`: Required Python packages

## Requirements

To run this code, you will need Python 3.8+ and the packages listed in `requirements.txt`. Install them with:

```bash
pip install -r requirements.txt
```

## Data

This project uses the MIMIC-IV (version 2.2) database, which requires appropriate credentialing and access permissions. The data is not included in this repository due to privacy and licensing restrictions. To apply for access to the MIMIC-IV database, please visit [PhysioNet](https://physionet.org/content/mimiciv/2.2/).

## Usage

To run the complete workflow:

```bash
python main.py --cohort_path path/to/cohort_data.csv --dynamic_path path/to/dynamic_data.parquet --output_dir output
```

Options:
- `--cohort_path`: Path to cohort data CSV file
- `--dynamic_path`: Path to dynamic data Parquet file
- `--output_dir`: Directory to save all outputs
- `--random_state`: Random state for reproducibility (default: 42)

## Models

The analysis implements multiple machine learning models:
- Logistic Regression
- XGBoost
- LightGBM
- AdaBoost
- Random Forest
- Gradient Boosting
- Gaussian Naive Bayes
- Complement Naive Bayes
- Multilayer Perceptron
- Support Vector Machine

Additionally, Cox proportional hazards regression is used for survival analysis.

## Interpretability

SHAP (SHapley Additive exPlanations) values are used to provide interpretable insights into model predictions, helping identify key predictive factors for mortality risk in immunocompromised ICU patients.

## Citation

If you use this code in your research, please cite our paper:

```
[Citation information will be added upon publication]
```
