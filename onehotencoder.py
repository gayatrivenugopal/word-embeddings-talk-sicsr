
"""
Sources:
https://machinelearningmastery.com/how-to-one-hot-encode-sequence-data-in-python/
https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html
https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html
"""

from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

sentence = "We are attending a virtual session"
words = sentence.split()

# label encoder - encodes the labels with a value between 0 and number of classes - 1
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(words)
print("After label encoding: ", integer_encoded)

# binary encoding
onehot_encoder = OneHotEncoder(sparse = False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
print("After reshaping: ")
print(integer_encoded)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
print("After one hot encoding: ")
print(onehot_encoded)
