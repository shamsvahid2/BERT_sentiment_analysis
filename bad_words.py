import pandas as pd
import numpy as np
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.feature_extraction.text import CountVectorizer

# model definition
def build_model():
    input_ids = Input(shape=(128,), dtype=tf.int32, name="input_ids")
    bert_model = TFBertModel.from_pretrained('bert-base-uncased', from_pt=False)
    bert_output = bert_model(input_ids)[1]

    output = Dense(1, activation='sigmoid')(bert_output)

    model = Model(inputs=input_ids, outputs=output)
    model.compile(optimizer=Adam(learning_rate=5e-5), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# load the model
model = build_model()
model.load_weights('fine_tuned_model/bert_finetuned_model.h5')

# load the dataset
data = pd.read_csv('data/test_preprocessed.csv')

# prepare data for prediction
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# tokenization and encoding
def encode_sentences(sentences):
    return tokenizer(sentences, padding=True, truncation=True, max_length=128, return_tensors="tf")
X = encode_sentences(data['sentences'].tolist())

# predict sentiments
predictions = model.predict( X['input_ids'])
data['prediction'] = (predictions.flatten() > 0.5).astype(int)  # Assuming sigmoid output, threshold at 0.5

# filter negative sentences
negative_sentences = data[data['prediction'] == 0]['sentences']

# count words across all negative sentences
th=15
vectorizer = CountVectorizer(stop_words='english', ngram_range=(1, 1))
neg_matrix = vectorizer.fit_transform(negative_sentences)
freqs = zip(vectorizer.get_feature_names_out(), neg_matrix.sum(axis=0).tolist()[0])
bad_words = {word for word, freq in freqs if freq >= th}

# extract bad words based on the frequency threshold
def extract_bad_words(sentence):
    words = set(sentence.split())
    return ', '.join(words.intersection(bad_words))

# populate 'Bad Words'
data['Bad Words'] = data.apply(lambda row: extract_bad_words(row['sentences']) if row['prediction'] == 0 else '', axis=1)
data.to_csv('data/bad-words_test_preprocessed.csv')
