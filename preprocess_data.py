import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split

# Loading data
def load_data(file_path):
    return pd.read_csv(file_path)

# Preprocessing text data
def preprocess_text(text):
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text.lower())
    filtered_words = [word for word in tokens if word not in stop_words]
    return " ".join(filtered_words)

# Main function to load, preprocess, and save the processed data
def main():
    train = load_data('data/train.csv')
    test = load_data('data/test.csv')

    train['sentences'] = train['sentences'].apply(preprocess_text)
    test['sentences'] = test['sentences'].apply(preprocess_text)

    train.to_csv('data/train_preprocessed.csv', index=False)
    test.to_csv('data/test_preprocessed.csv', index=False)

if __name__ == '__main__':
    main()
