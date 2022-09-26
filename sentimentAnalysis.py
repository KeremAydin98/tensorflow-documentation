import tensorflow as tf
import shutil, string


"""
Load the dataset
"""
url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
file_dir = tf.keras.utils.get_file(fname="aclImdb_v1.tar.gz",origin=url, untar=True, cache_dir=".")

shutil.rmtree("/content/datasets/aclImdb/train/unsup", ignore_errors=True)

dataset_dir = "datasets/aclImdb/train"

train_ds = tf.keras.utils.text_dataset_from_directory(dataset_dir, batch_size=32, validation_split=0.2, subset="training", seed=42)

val_ds = tf.keras.utils.text_dataset_from_directory(dataset_dir, batch_size=32, validation_split=0.2, subset="validation", seed=42)

"""
Speed up the training by using .cache and .prefetch methods
"""
train_ds = train_ds.cache().prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.cache().prefetch(tf.data.AUTOTUNE)


"""
Preprocessing text to remove unnecessary characters
"""
def preprocess_text(text):

  text = tf.strings.lower(text)
  text = tf.strings.regex_replace(text, "<br />"," ")

  return tf.strings.regex_replace(text, '[%s]' % string.punctuation, "")


vectorize_layer = tf.keras.layers.TextVectorization(10000,
                                                    standardize=preprocess_text,
                                                    output_sequence_length=1000)

"""
Create the sentiment analysis model
"""
text_ds = train_ds.map(lambda x,y:x)

vectorize_layer.adapt(text_ds)

model = tf.keras.Sequential([
    vectorize_layer,
    tf.keras.layers.Embedding(10000, 5),
    tf.keras.layers.GlobalMaxPool1D(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

model.compile(loss = tf.keras.losses.binary_crossentropy,
              optimizer = tf.keras.optimizers.Adam(),
              metrics=["accuracy"])

model.fit(train_ds, validation_data = val_ds, epochs=10)


"""
Evaluating the model
"""
test_directory = "datasets/aclImdb/test"
test_ds = tf.keras.utils.text_dataset_from_directory(test_directory, batch_size=32)


def trim_text(text):
    return str(text)[:1000]


labels = ["Negative", "Positive"]

for tests, results in test_ds.take(1):

    for i, test in enumerate(tests):
        test = trim_text(test)
        print(f"Sentence:\n{str(test)}")
        print(f"Prediction:\n{labels[int(tf.round(model.predict([test])))]}")
        print(f"Correct label:\n{labels[results[i]]}")
        print("-----------------------\n")

