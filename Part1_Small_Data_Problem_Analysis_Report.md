# Small Data Problem Analysis Report

## Scenario 1: Travel Planner

**Recommended strategy: Transfer learning.** The input is a small labeled image dataset with three categories: forest, desert, and beach. Training a visual model from scratch with fewer than 500 examples would likely overfit. A model such as VGG16 has already learned general visual features (edges, textures, shapes, and object parts) from a much larger image collection. Keep those learned layers fixed and train a new final classification layer to map those features to the three travel categories. Image augmentation can provide additional variation during training, while validation and test images should use deterministic resizing and cropping.

## Scenario 2: Loan Funding Prediction

**Recommended strategy: Synthetic data generation.** The input is structured, tabular loan data, and the company already has a fairly large dataset. Its main problem is class imbalance: there are too few denied-loan examples. A Variational Autoencoder (VAE) can be trained on the underrepresented denied class to learn its feature distribution and sample additional plausible records. Add those generated rows to the original dataset, assign them the denied label, and compare class-specific precision, recall, and F1 scores before and after augmentation. Generated records should be checked for validity and evaluated on a held-out set so improvements reflect generalization rather than simply reproducing training data.

## Strategy Summary

Transfer learning reuses visual knowledge learned from a large external image dataset and adapts the final classifier to a small image task. Synthetic data generation learns patterns within the underrepresented class of a tabular dataset and creates more examples for model training. The strategies fit the data type and the specific small-data problem in each scenario.
