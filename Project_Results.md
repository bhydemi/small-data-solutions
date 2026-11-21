# Project Results

## Confirmed meaning of the loan label

The supplied CSV has 67,463 rows and the same loan fields as the source dataset. The source dataset description identifies this dataset as a **loan-default prediction** dataset and defines `Loan Status = 1` as **defaulter** and `0` as **non-defaulter**; it cites the [Kaggle Deloitte Hackathon dataset](https://www.kaggle.com/datasets/sachinsarkar/deloitte-hackathon). A [dataset mirror and feature description](https://github.com/Shreya-Rawat/Bank-Loan-Defaulter-Prediction) states the same label mapping and gives the matching 67,463-row count.

So the prompt's description of class 1 as a denied application does **not** match this CSV. It represents default outcome among loans, not whether an application was rejected. The data also contains post-funding fields such as total interest received, late fees, recoveries, collection fees, and payment history; those are not valid predictors for an approval decision made before funding and can leak outcome information into a default model. The current VAE workflow still follows the assignment's mechanical instruction to augment rows where `Loan Status = 1`, but those rows should be described as defaults, not denied applications.

## Part 2: Travel image classifier

- Model: ImageNet-pretrained VGG16 with frozen pretrained layers and a three-class replacement head.
- Training: 8 epochs; best validation accuracy was 98.3%.
- Held-out test accuracy: **97.9% (47/48 images)**. Beach: 16/16, desert: 15/16, forest: 16/16.
- The saved checkpoint contains the task-specific classifier head and class names. The pretrained VGG16 backbone is retrieved by torchvision when rebuilding the model.

## Part 3: Loan data augmentation

The VAE used a 10-dimensional latent space, larger hidden layers, and a lower KL weight; it trained for 750 epochs on the 6,241 denied-loan records and generated 50,000 additional rows. The median generated-to-real feature standard-deviation ratio was 0.81 (10th to 90th percentile: 0.74 to 0.90), up from severe under-dispersion with the starter settings. Evaluation reserved real loan records for calibration and testing; generated rows were included only in augmented training. The decision threshold was selected on the calibration split to maximize denied-class F1.

| Held-out metric | Original data | With synthetic rows |
| --- | ---: | ---: |
| Accuracy | 0.093 | 0.141 (12,500 synthetic rows) |
| Class 0 precision / recall / F1 | 0.90 / 0.00 / 0.00 | 0.91 / 0.06 / 0.11 |
| Class 1 precision / recall / F1 | 0.09 / 1.00 / 0.17 | 0.09 / 0.94 / 0.17 |
| Macro F1 | 0.09 | 0.14 |

Using all 50,000 synthetic rows gave macro F1 0.09, so the smaller synthetic ratio performed better on macro F1. However, denied-loan (class 1) F1 stayed at 0.17 at both ratios. Class 1 precision remains low, so these results do **not** establish a useful denied-loan predictor. The VAE now produces more varied rows, but the classifier still does not distinguish denied loans well on held-out real data.

The comparison uses a fixed real-data split (65% training, 15% threshold calibration, 20% final test) with seed 42/43. The threshold is calibrated separately for each model on real calibration examples. Because the dataset is highly imbalanced and the classifier's scores are poorly calibrated, accuracy alone is not informative here.

## Predictive-signal check

`starter_code/part2-synthetic/model_diagnostics.py` compares class-weighted logistic regression, random forest, and histogram gradient boosting on that same real-only split. ROC-AUC was 0.508–0.525 and PR-AUC was 0.096–0.102 on original data; the positive-class prevalence is 0.093. These scores are close to a no-skill ranking. With 25% of the VAE rows, random-forest ROC-AUC was 0.529, PR-AUC 0.103, and class-1 F1 0.171, versus 0.515, 0.098, and 0.168 on original data. Histogram boosting's class-1 F1 changed from 0.167 to 0.171. These gains are small and do not establish a dependable denied-loan predictor.

Run the classifier comparison with `python starter_code/part2-synthetic/model_diagnostics.py`. The next meaningful improvement depends on finding better predictive features or labels; additional synthetic rows alone are unlikely to solve this dataset's weak-signal problem.

## Label and feature audit

`starter_code/part2-synthetic/feature_signal_report.py` found 67,463 rows, valid binary labels (0: 61,222; 1: 6,241), no missing values, and no duplicate feature rows with conflicting labels. The strongest single feature had only 0.537 test ROC-AUC. Random forest ROC-AUC was 1.000 on training data and 0.527 on held-out data; histogram boosting was 0.622 and 0.527. This gap indicates substantial overfitting, while the held-out scores show little generalizable signal in the available features. Run the audit with `python starter_code/part2-synthetic/feature_signal_report.py`.
