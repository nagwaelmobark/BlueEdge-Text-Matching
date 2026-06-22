# BlueEdge Text Matching

BlueEdge Text Matching is a lightweight machine learning pipeline for text matching and duplicate name detection. The project evaluates classical similarity features and machine learning classifiers to identify whether two text records refer to the same entity.

## Project Overview

This repository contains the experimental code for duplicate detection using similarity-based feature engineering and machine learning models. The pipeline includes preprocessing, feature extraction, group-based train/test splitting, model evaluation, cross-validation, ablation analysis, and benchmark comparisons.

## Main Features

* Text normalization and preprocessing
* Levenshtein similarity
* Jaro-Winkler similarity
* Fuzzy ratio similarity
* Token-sort ratio similarity
* Length ratio similarity
* TF-IDF character n-gram cosine similarity
* Group-based train/test split to reduce data leakage
* Machine learning model comparison:

  * Logistic Regression
  * Support Vector Machine
  * Random Forest
  * K-Nearest Neighbors
* SBERT feature comparison
* Cross-validation experiments
* Ablation study
* Benchmark comparison against spaCy and TextBlob

## Repository Structure

```text
BlueEdge-Text-Matching/
│
├── text_matching_and_duplicate_detectionv2.py
├── README.md
├── requirements.txt
├── .gitignore
└── LICENSE
```

## Installation

Install the required Python packages:

```bash
pip install -r requirements.txt
```

For spaCy benchmark experiments, also run:

```bash
python -m spacy download en_core_web_sm
```

## Usage

The original code was developed in Google Colab. To run it locally, replace Colab-specific upload and download commands with local file paths.

Example:

```python
df = pd.read_csv("data/final_dataset.csv", encoding="utf-8-sig")
```

Then run the main script:

```bash
python text_matching_and_duplicate_detectionv2.py
```

## Dataset

The dataset is not included in this repository. Place your dataset in a local `data/` folder and update the file path in the script before running the experiments.

## Outputs

The pipeline can generate:

* Model comparison results
* Cross-validation scores
* Per-category recall analysis
* Ablation study tables
* Benchmark results
* Figures and tables for reporting

## Notes

Some sections of the script are designed for Google Colab and may require minor changes before running locally, especially:

* `files.upload()`
* `files.download()`
* `/content/` paths
* Notebook-style package installation commands

## License

This project is released for academic and research purposes.
