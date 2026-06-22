# BlueEdge Text Matching

BlueEdge Text Matching is a lightweight machine learning pipeline for text matching and duplicate name detection. The project uses classical similarity features and machine learning models to identify whether two text records refer to the same entity.

## Project Overview

This repository contains the experimental workflow for duplicate detection using similarity-based feature engineering and machine learning. The code was originally developed in Google Colab and then reorganized into modular Python files for better readability and reproducibility.

The pipeline includes:

* Text preprocessing and normalization
* Similarity feature extraction
* Group-based train/test splitting to reduce data leakage
* Machine learning model comparison
* Cross-validation experiments
* Ablation analysis
* Benchmark comparison against spaCy and TextBlob

## Features Used

The project extracts the following similarity features:

* Levenshtein similarity
* Jaro-Winkler similarity
* Fuzzy ratio
* Token-sort ratio
* Length ratio
* TF-IDF character n-gram cosine similarity

## Machine Learning Models

The following classifiers are evaluated:

* Logistic Regression
* Support Vector Machine
* Random Forest
* K-Nearest Neighbors

## Repository Structure

```text
BlueEdge-Text-Matching/
│
├── notebooks/
│   └── text_matching_and_duplicate_detectionV2.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── features.py
│   ├── train_models.py
│   ├── evaluation.py
│   └── benchmark.py
│
├── results/
│   ├── figures/
│   └── tables/
│
├── data/
│   └── README.md
│
├── requirements.txt
├── README.md
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

## Dataset

The dataset is not included in this repository.

Place your dataset inside the `data/` folder and update the dataset path in the script or notebook.

Expected columns include:

* `pair_id`
* `group_id_1`
* `group_id_2`
* `name_1`
* `name_2`
* `is_duplicate`
* `error_type`
* `language`

## Usage

Open the notebook:

```text
notebooks/text_matching_and_duplicate_detectionV2.ipynb
```

Or import the reusable modules from the `src/` folder:

```python
from src.preprocessing import add_normalized_columns
from src.features import add_similarity_features
from src.train_models import get_default_models
from src.evaluation import evaluate_predictions
```

## Results

Generated figures and tables should be saved in:

```text
results/figures/
results/tables/
```

These outputs are not included by default and can be reproduced by running the experiment scripts or notebook.

## Notes

Some parts of the original experiment were designed for Google Colab. The modular files in `src/` are intended to make the project easier to reuse, maintain, and reproduce.

## License

This project is licensed under the MIT License.
