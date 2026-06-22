# Data Folder

This folder is reserved for the dataset used in the BlueEdge text matching experiments.

## Dataset Status

The dataset is not included in this repository to avoid sharing private, sensitive, or large data files.

## Expected Dataset Format

The main script expects a CSV file with columns similar to:

```text
Usage

Place your dataset file in this folder, for example:

data/final_blueedge_dataset.csv

Then update the dataset path in the script or notebook:

df = pd.read_csv("data/final_blueedge_dataset.csv", encoding="utf-8-sig")
Notes

Do not upload private or confidential datasets unless they are properly anonymized and approved for public release.
pair_id
group_id_1
group_id_2
name_1
name_2
is_duplicate
error_type
language
