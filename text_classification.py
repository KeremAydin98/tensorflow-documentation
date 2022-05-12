"""
This tutorial demonstrates text classification starting from plain text
files stored on disk. You'll train a binary classifier to perform sentiment
analysis on an IMDB dataset. At the end of the notebook,
there is an exercise for you to try, in which you'll train a multi-class
classifier to predict the tag for a programming question on Stack Overflow.
"""

import matplotlib.pyplot as plt
import os
import re
import shutil
import string
import tensorflow as tf
import pandas as pd

# Download url
url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"

# Download file from this url
dataset = tf.keras.utils.get_file("aclImdb_v1", url,untar=True, cache_dir=".",
                                  cache_subdir='')

# Assign directories
dataset_dir = os.path.join(os.path.dirname(dataset),"aclImdb")
train_dir = os.path.join(dataset_dir, "train")

# Lets see a single sample of a movie review
sample_file = os.path.join(train_dir, 'pos/1181_9.txt')
with open(sample_file) as f:
    print(f.read())

remove_dir = os.path.join(train_dir, 'unsup')
shutil.rmtree(remove_dir)

# Configurations
BATCH_SIZE = 32
SEED = 42

train_dataset = tf.keras.utils.text_dataset_from_directory(directory=train_dir,
                                                           batch_size=BATCH_SIZE,
                                                           validation_split=0.2,
                                                           seed=SEED,
                                                           subset="training")

validation_dataset = tf.keras.utils.text_dataset_from_directory(directory=train_dir,
                                                           batch_size=BATCH_SIZE,
                                                           seed = SEED,
                                                           validation_split=0.2,
                                                           subset="validation")

test_dataset = tf.keras.utils.text_dataset_from_directory(directory="aclImdb/test",
                                                          batch_size=BATCH_SIZE)

# Lets print 3 examples before the training
for text_batch, label_batch in train_dataset.take(1):
    print("******THREE SAMPLES OF TEXT AND LABELS******")
    for i in range(3):
        print("Review:",text_batch.numpy()[i])
        print("Label:",label_batch.numpy()[i])

# Standardize and tokenize the dataset
def custom_standardization(input_data):

    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, '<br />',' ')
    return tf.strings.regex_replace(stripped_html,
                                    '[%s]' % re.escape(string.punctuation),"")

# Vectorization refers to converting tokens into numbers so they can be fed into a neural network.
max_vocabulary = 10000
sequence_length = 250

vectorization_layer = tf.keras.layers.TextVectorization(standardize=custom_standardization,
                                                        max_tokens=max_vocabulary,
                                                        output_mode="int",
                                                        output_sequence_length=sequence_length)

# Next, you will call adapt to fit the state of the preprocessing layer to the dataset.
# Text only dataset
train_text = train_dataset.map(lambda x, y:x)
vectorization_layer.adapt(train_text)

# Lets test it
for text, label in train_dataset.take(1):

    print("******TEST OF VECTORIZATION LAYER******")
    print(f"Text: {text[0]}")
    print(f"Label: {label[0]}")
    print(f"Vectorized version {vectorization_layer(text[0])}")

# Since we cannot save the model which has a vectorization layer, we need to transform
# our dataset
def vectorize_text(text, label):
  text = tf.expand_dims(text, -1)
  return vectorization_layer(text), label


train_dataset = train_dataset.map(vectorize_text)
validation_dataset = validation_dataset.map(vectorize_text)
test_dataset = test_dataset.map(vectorize_text)

# cache() keeps data in memory after it's loaded off disk.
# .prefetch() overlaps data preprocessing and model execution while training
train_dataset = train_dataset.cache().prefetch(tf.data.AUTOTUNE)
validation_dataset = validation_dataset.cache().prefetch(tf.data.AUTOTUNE)
test_dataset = test_dataset.cache().prefetch(tf.data.AUTOTUNE)

# Create the model
embedding_dim = 16
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(max_vocabulary + 1, embedding_dim),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
              optimizer=tf.keras.optimizers.Adam(),
              metrics=["accuracy"])

history = model.fit(train_dataset, epochs=10, validation_data=validation_dataset)

pd.DataFrame(history.history).plot()
plt.show()

model.save("text_classification.h5")






