# Small Data Strategies: Transfer Learning and Synthetic Data

This repository contains a three-part small-data project: strategy selection, transfer learning for travel images, and VAE-based tabular augmentation.

## Important loan-label clarification

The supplied 67,463-row loan dataset is a default-prediction dataset. Its source defines `Loan Status = 1` as **defaulter** and `0` as **non-defaulter**; it does not indicate whether a loan application was denied. The dataset comes from the [Deloitte Hackathon loan dataset on Kaggle](https://www.kaggle.com/datasets/sachinsarkar/deloitte-hackathon). Some columns record post-funding events, so they are not appropriate predictors for an application-time approval decision. The scripts augment the status-1 class as required by the project instructions, but results should be interpreted as default-class results. Details are in [Project_Results.md](Project_Results.md).

## Results

- Travel image classifier: **97.9% accuracy** on 48 held-out images, with a best validation accuracy of 98.3%.
- Loan experiment: the VAE generates 50,000 status-1 rows. A 25% synthetic-data ratio gave only a small class-1 F1 gain (0.168 to 0.171) with a class-weighted random forest. Held-out ROC-AUC remains close to chance, so this dataset does not support a reliable predictor.
- See [Project_Results.md](Project_Results.md) for split design, class metrics, label audit, and limitations.

## Setup

Use Python 3.10 or later. Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

The image script downloads pretrained VGG16 weights from torchvision the first time it runs.

## Run the project

Train and evaluate the travel classifier:

```bash
python starter_code/part1-transfer/transfer_learning.py
```

Train the VAE on `Loan Status = 1`, generate 50,000 synthetic rows, write the augmented CSV, and evaluate the baseline and two synthetic-data ratios:

```bash
python starter_code/part2-synthetic/synthetic_data.py
```

Run the held-out classifier comparison or inspect labels and feature signal:

```bash
python starter_code/part2-synthetic/model_diagnostics.py
python starter_code/part2-synthetic/feature_signal_report.py
```

## Submission files

- [Part 1 analysis report (PDF)](Part1_Small_Data_Problem_Analysis_Report.pdf)
- [Part 1 analysis report (Markdown)](Part1_Small_Data_Problem_Analysis_Report.md)
- [Measured results and data interpretation](Project_Results.md)
- Trained travel classifier head: `starter_code/part1-transfer/travel_vgg16.pth`
- Augmented loan data: `starter_code/part2-synthetic/data/loan_continuous_expanded.csv`

No license file is included.
