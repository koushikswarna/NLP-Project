# Yelp Review Sentiment Analysis

This repository contains a project for analyzing Yelp reviews and performing a binary sentiment classification (1-star vs 5-star reviews) using Natural Language Processing (NLP) techniques in Python.

## Overview

The project includes:

1. **Exploratory Data Analysis (EDA)**  
   - Examining the structure of the Yelp dataset.
   - Visualizing review text lengths, star distributions, and correlations between numeric features.
   - Using histograms, boxplots, countplots, and heatmaps to understand patterns in the data.

2. **Text Classification**  
   - Filtering the dataset to only 1-star and 5-star reviews for binary sentiment analysis.
   - Converting text into numerical representations using `CountVectorizer` and `TfidfTransformer`.
   - Training a `Multinomial Naive Bayes` model for sentiment classification.

3. **Pipeline and Model Evaluation**  
   - Using scikit-learn’s `Pipeline` to combine preprocessing and model training in one step.
   - Evaluating the model with `classification_report` and `confusion_matrix`.

## Dataset

- The dataset `yelp.csv` contains Yelp reviews with at least the following columns:  
  - `text`: The content of the review.  
  - `stars`: The rating associated with the review (1–5).

- For classification purposes, only 1-star and 5-star reviews were considered.

## Requirements

Python packages used:

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
