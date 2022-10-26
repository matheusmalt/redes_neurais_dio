import os
from xml.etree.ElementInclude import include
import matplotlib.pyplot as plt
import tensorflow as tf

dataset_dir = os.path.join(os.getcwd(), "cats_and_dogs_filtered")
dataset_train_dir = os.path.join(dataset_dir, "train")
dataset_train_cats_len = len(os.listdir(os.path.join(dataset_train_dir, "cats")))
dataset_train_dogs_len = len(os.listdir(os.path.join(dataset_train_dir, "dogs")))

dataset_validation_dir = os.path.join(dataset_dir, "validation")
dataset_validation_cats_len = len(os.listdir(os.path.join(dataset_validation_dir, "cats")))
dataset_validation_dogs_len = len(os.listdir(os.path.join(dataset_validation_dir, "dogs")))

print("Train Cats: {}".format(dataset_train_cats_len))
print("Train Dogs: {}".format(dataset_train_dogs_len))
print("Validation Cats: {}".format(dataset_validation_cats_len))
print("Validation Dogs: {}".format(dataset_validation_dogs_len))

image_width = 160
image_height = 160

image_color_channel = 3
image_color_channel_size = 255
image_size = (image_width, image_height)
image_shape = image_size + (image_color_channel,)

batch_size = 32
epochs = 20
learning_rate = 0.0001
class_name = ["cat", "dog"]

dataset_train = tf.keras.preprocessing.image_dataset_from_directory(
    dataset_train_dir,
    image_size = image_size,
    batch_size = batch_size,
    shuffle = True
)

dataset_validation = tf.keras.preprocessing.image_dataset_from_directory(
    dataset_validation_dir,
    image_size = image_size,
    batch_size = batch_size,
    shuffle = True
)

dataset_validation_cardinality = tf.data.experimental.cardinality(dataset_validation)
dataset_validation_batches = dataset_validation_cardinality // 5
dataset_test = dataset_validation.take(dataset_validation_batches)
dataset_validation = dataset_validation.skip(dataset_validation_batches)

def plot_dataset(dataset):
    plt.gcf().clear()
    plt.figure(figsize = (15, 15))

    for features, labels in dataset.take(1):
        for i in range(9):
            plt.subplot(3, 3, i + 1)
            plt.axis("off")
            plt.imshow(features[i].numpy().astype("uint8"))
            plt.title(class_name[labels[i]])

plot_dataset(dataset_train)
plot_dataset(dataset_validation)
plot_dataset(dataset_test)

data_augmentation = tf.keras.models.Sequential([
    tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal"),
    tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
    tf.keras.layers.experimental.preprocessing.RandomZoom(0.2)
])

def plot_dataset_data_augmentation(dataset):
    plt.gcf().clear()
    plt.figure(figsize= (15, 15))

    for features, _ in dataset.take(1):
        
        feature = features[0]

        for i in range(9):

            feature_data_argumentation = data_augmentation(tf.expand_dims(feature, 0))

            plt.subplot(3, 3, i + 1)
            plt.axis("off")

            plt.imshow(feature_data_argumentation[0 / image_color_channel_size])

plot_dataset_data_augmentation(dataset_train)

model_transfer_learning = tf.keras.applications(
    input_shape = image_shape,
    include_top = False,
    weights = "imagenet"
)

model_transfer_learning.trainable = False
model_transfer_learning.summary()

model = tf.keras.models.Sequential([
    tf.keras.layers.experimental.preprocessing.Rescaling(
        1. / image_color_channel_size,
        input_shape = image_shape
    ),
    data_augmentation,
    model_transfer_learning,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation = "sigmoid")
])

model.compile(
    optimizer= tf.keras.optimizers.Adam(learning_rate = learning_rate),
    loss = tf.keras.losses.BinaryFocalCrossentropy(),
    metrics = ["accuracy"]
)

model.summary()

history = model.fit(
    dataset_train,
    validation_data = dataset_validation,
    epochs = epochs
)

def plot_model():
    accuracy = history.history["accuracy"]
    val_accuracy = history.history["val_accuracy"]

    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    epochs_range = range(epochs)

    plt.gcf().clear()
    plt.figure(figsize = (15, 8))

    plt.subplot(1, 2, 1)

    plt.title("Training and Validation Accuracy")
    plt.plot(epochs_range, accuracy, label = "Training Accuracy")
    plt.plot(epochs_range, val_accuracy, label = "Validation Accuracy")
    plt.legend(loc = "lower right")

    plt.subplot(1, 2, 2)
    plt.title("Training and Validation Loss")
    plt.plot(epochs_range, loss, label = "Training Loss")
    plt.plot(epochs_range, val_loss, label = "Validation Loss")
    plt.legend(loc = "lower right")

    plt.show()

plot_model()

def plot_dataset_predections(dataset):
    features, labels = dataset.as_numpy_iterator().next()
    predictions = model.predict_on_batch(features).flatten()
    predictions = tf.where(predictions < 0.5, 0, 1)

    print("Labels: {}".format(labels))
    print("Predictions: {}".format(predictions.numpy()))

    plt.gcf().clear()
    plt.figure(figsize = (15, 15))

    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.axis("off")
        plt.imshow(features[i].astype("uint8"))
        plt.title(class_name[predictions[i]])

plot_dataset_predections(dataset_test)

model.save("path/to/model")

model = tf.keras.models.load_model("path/to/model")