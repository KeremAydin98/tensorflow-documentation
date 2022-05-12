import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub

# Split the training set into 60% and 40% to end up with 15000 examples for training
train_data, validation_data, test_data = tfds.load(name="imdb_reviews",
                                                   split=("train[:60%]","train[60%:]","test"),
                                                   as_supervised=True)
# Lets inspect the data
for train_text, train_label in train_data.take(1):

    print(f"Text: {train_text}")
    print(f"Label: {train_label}")

# One way to represent the text is to convert sentences into embeddings vectors
# Larger dimensional embeddings can improve on your task but it may take longer to train your model.

embedding = "https://tfhub.dev/google/nnlm-en-dim50/2"
hub_layer = hub.KerasLayer(embedding,trainable=True, dtype=tf.string,input_shape=[])

# Print one example and its embeddings
train_text_batch, train_label_batch = next(iter(train_data.batch(1)))

print(f"Text: {train_text_batch}")
print(f"Label: {train_label_batch}")
print(f"Embedded version: {hub_layer(train_text_batch)}")

# Lets now build the model
model = tf.keras.Sequential([
    hub_layer,
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

# inary_crossentropy is better for dealing with probabilities—it measures the "distance" between probability distributions, or in our case,
# between the ground-truth distribution and the predictions.
model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
              optimizer=tf.keras.optimizers.Adam(),
              metrics=["accuracy"])


history = model.fit(train_data.batch(512), epochs=10, validation_data=validation_data.batch(512))

results = model.evaluate(validation_data.batch(512))

pd.DataFrame(history.history).plot(figsize=(10,10))
plt.show()

model.save("text_classification_with_hub.h5")


