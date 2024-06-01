import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from transformers import TFBertModel, BertTokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

import pandas as pd
import numpy as np

# load the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# load the dataset
train = pd.read_csv('data/train_preprocessed.csv')
test = pd.read_csv('data/test_preprocessed.csv')

# tokenization and encoding
def encode_sentences(sentences):
    return tokenizer(sentences, padding=True, truncation=True, max_length=128, return_tensors="tf")

# preparing the dataset
X_train, X_val, y_train, y_val = train_test_split(train['sentences'], train['labels'], test_size=0.1)
X_train_enc = encode_sentences(X_train.tolist())
X_val_enc = encode_sentences(X_val.tolist())
X_test_enc = encode_sentences(test['sentences'].tolist())

# model definition
def build_model():
    input_ids = Input(shape=(128,), dtype=tf.int32, name="input_ids")
    bert_model = TFBertModel.from_pretrained('bert-base-uncased', from_pt=False)
    bert_output = bert_model(input_ids)[1]

    output = Dense(1, activation='sigmoid')(bert_output)

    model = Model(inputs=input_ids, outputs=output)
    model.compile(optimizer=Adam(learning_rate=5e-5), loss='binary_crossentropy', metrics=['accuracy'])
    return model

model = build_model()

# training
print('training just started, please wait!')
model.fit(X_train_enc['input_ids'], y_train, validation_data=(X_val_enc['input_ids'], y_val), epochs=1, batch_size=8)
print('training just finished, evaluation is starting...')

# evaluating the model
predictions = model.predict(X_test_enc['input_ids'])
predictions = (predictions > 0.5).astype(int)
cm = confusion_matrix(test['labels'], predictions)
print("Confusion Matrix:\n", cm)
print(classification_report(test['labels'], predictions))

# save the model
model.save('fine_tuned_model/bert_finetuned_model.h5')
