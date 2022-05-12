"""
This tutorial uses the classic Auto MPG dataset and demonstrates how to build models to
predict the fuel efficiency of the late-1970s and early 1980s automobiles. To do this,
you will provide the models with a description of many automobiles from that time period.
This description includes attributes like cylinders, displacement, horsepower, and weight.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf

# Get the data
url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
                'Acceleration', 'Model Year', 'Origin']
dataset = pd.read_csv(url, names=column_names,na_values="?",comment="\t",
                      sep=" ",skipinitialspace=True)

# Drop the none values
dataset = dataset.dropna()

# The "Origin" column is categorical, not numeric.
dataset["Origin"] = dataset["Origin"].map({1:"USA",2:"Europa",3:"Japan"})
dataset = pd.get_dummies(dataset, columns=["Origin"],prefix="",prefix_sep="")

sns.pairplot(dataset[["MPG","Cylinders","Displacement","Weight"]],diag_kind='kde')
plt.show()

features = dataset.drop("MPG",axis=1)
labels = dataset["MPG"]

split_size = int(0.8 * len(features))
train_features, train_labels = features[:split_size], labels[:split_size]
test_features, test_labels = features[split_size:], labels[split_size:]

# The tf.keras.layers.Normalization is a clean and simple way to add feature normalization into your model.
horsepower = np.array(train_features['Horsepower'])

horsepower_normalizer = tf.keras.layers.Normalization(input_shape=[1,],axis=None)
horsepower_normalizer.adapt(np.array(horsepower))

# Create the model
linear_model = tf.keras.Sequential([
    horsepower_normalizer,
    tf.keras.layers.Dense(1)
])

linear_model.compile(loss=tf.keras.losses.mae,
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.1))

history = linear_model.fit(train_features["Horsepower"], train_labels, epochs=100, validation_split=0.2)

pd.DataFrame(history.history).plot(figsize=(10,10))
plt.show()

predictions = linear_model.predict(tf.linspace(0,250,251))
plt.scatter(train_features["Horsepower"], train_labels)
plt.plot(tf.linspace(0,250,251), predictions,color="black")
plt.show()

dl_model = tf.keras.Sequential([
    horsepower_normalizer,
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(1)
])

dl_model.compile(loss=tf.keras.losses.mae,
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.1))

dl_history = dl_model.fit(train_features["Horsepower"], train_labels, epochs=100, validation_split=0.2)

pd.DataFrame(dl_history.history).plot(figsize=(10,10))
plt.show()

predictions = dl_model.predict(tf.linspace(0,250,251))
plt.scatter(train_features["Horsepower"], train_labels)
plt.plot(tf.linspace(0,250,251), predictions,color="black")
plt.show()


