import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist

print(tf.__version__)

(x_train, y_train),(x_test, y_test) = fashion_mnist.load_data()

class_names = ["T-shirt/top","Trouser","Pullover","Dress","Coat","Sandal","Shirt",
              "Sneaker","Bag","Ankle boot"]

plt.figure(1)
plt.imshow(x_train[0])
plt.title("One sample")
plt.grid(False)

# Preprocess data
x_train = x_train / 255
x_test = x_test / 255

plt.figure(2,figsize=(10,10))
for i in range(25):
    plt.suptitle("25 samples")
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_train[i])
    plt.xlabel(class_names[y_train[i]])

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(len(class_names),activation="softmax")
])

model.compile(loss=tf.keras.losses.sparse_categorical_crossentropy,
              optimizer=tf.keras.optimizers.Adam(),
              metrics=["accuracy"])

model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10)

loss, accuracy = model.evaluate(x_test, y_test)

print(f"Final accuracy: {accuracy}")
print(f"Final validation loss: {loss}")

predictions = model.predict(x_test)

figure, ax = plt.subplots(1, 5, figsize=(10,10))
for i in range(5):

    ax[i].imshow(x_test[i])
    ax[i].grid(False)
    if y_test[i] == np.argmax(predictions[i]):
        c = "green"
    else:
        c = "red"
    ax[i].set_title(f"True class: {class_names[y_test[i]]}\nPred class:{class_names[np.argmax(predictions[i])]}",
               color=c)

figure.tight_layout()
plt.show()

model.save("basic_classification_model.h5")