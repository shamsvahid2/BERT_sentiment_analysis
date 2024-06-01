# Advanced Sentiment Analysis for Customer Reviews

## Project Overview

This project implements an advanced sentiment analysis model for customer reviews using BERT and TensorFlow/Keras. The model classifies customer reviews as positive or negative and identifies inappropriate language in negative reviews. The following steps outline the process:

1. **Data Preprocessing**: Clean and prepare the dataset.
2. **Model Training and Evaluation**: Fine-tune a pre-trained BERT model and evaluate its performance.
3. **Bad Words Identification**: Identify inappropriate language in negative reviews.

## Setup Instructions

### Create a Conda Environment and Install Requirements

1. **Create a new Conda environment:**

    ```bash
    conda create -n sentiment_analysis python=3.8
    ```

2. **Activate the environment:**

    ```bash
    conda activate sentiment_analysis
    ```

3. **Install the required packages:**

    ```bash
    pip install -r requirements.txt
    ```

### Scripts Description

#### 1. `preprocess_data.py`

This script preprocesses the text data by performing the following steps:

- Removing HTML tags.
- Eliminating punctuation.
- Removing stopwords.
- Tokenizing the sentences.

The script processes both training and test datasets and saves the preprocessed data as `train_preprocessed.csv` and `test_preprocessed.csv`.

**Usage:**

```bash
python preprocess_data.py
```

#### 2. `train_evaluate_model.py`

This script trains and evaluates a BERT model:

- Splits the preprocessed training data into training and validation sets.
- Fine-tunes a pre-trained BERT model on the training data.
- Evaluates the model using a confusion matrix on the test data.
- Saves the fine-tuned model.

**Usage:**

```bash
python train_evaluate_model.py
```

#### 3. `bad_words.py`

This script performs the following steps:

- Loads the fine-tuned BERT model and the preprocessed test data.
- Predicts the sentiment of the test reviews.
- Identifies negative reviews and extracts bad words based on a frequency threshold. (This is a very naive and we can make it really better)

**Usage:**

```bash
python bad_words.py
```

#### 4. `EDA.ipynb`

This Jupyter notebook performs Exploratory Data Analysis (EDA) on the dataset. It also converts the data type to CSV format for further processing.

### Model Download
Due to the high volume of the fine-tuned model, it is uploaded to Google Drive. Download the model from the following link:
[Download Fine-Tuned BERT Model](https://drive.google.com/drive/folders/1SBRwXhecmY_vPhdTTnZrwj_YjGRtIR1D?usp=sharing)
Please put it on `fine_tuned_model` folder so you can run `bad_words.py`.

### Requirements
Ensure you install `requirements.txt`

## Classification Report

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0     | 0.90      | 0.90   | 0.90     | 1951    |
| 1     | 0.90      | 0.91   | 0.91     | 2049    |

**Accuracy**: 0.90  
**Macro Average**: Precision: 0.90, Recall: 0.90, F1-Score: 0.90  
**Weighted Average**: Precision: 0.90, Recall: 0.90, F1-Score: 0.90  
**Total Support**: 4000

## Future work:

As I don't have much time for this task I choose a naive way to extract bad words which we can make it really better with other kinds of transformer based methods.
