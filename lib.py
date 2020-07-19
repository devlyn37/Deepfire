import tensorflow as tf
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import matplotlib
matplotlib.use('pdf')


def createModel(baseModel, hidden_layers, num_classes):
    model = Sequential()

    # Transfer Learning
    model.add(baseModel)
    model.layers[0].trainable = False

    # Hidden Layers
    for layer in range(len(hidden_layers)):
        model.add(Dense(hidden_layers[layer], activation='relu'))

    # Add our classification layer and display model properties
    model.add(Dense(num_classes, activation='softmax'))
    model.summary()

    # Compile the sections into one NN
    model.compile(optimizer='sgd', loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def trainModel(datasetPath, model, epochs, batch_size, image_size, preprocess_input):
    data_generator = ImageDataGenerator(
        preprocessing_function=preprocess_input)

    train_generator = data_generator.flow_from_directory(
        f'{datasetPath}/train',
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode='categorical')
    num_training_files = len(train_generator.filepaths)

    validation_generator = data_generator.flow_from_directory(
        f'{datasetPath}/validate',
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode='categorical')
    num_validation_files = len(validation_generator.filepaths)

    # The division by 5 here is to make training quicker
    # when you want maximum accuracy remove it.
    return model.fit(
        train_generator,
        epochs=epochs,
        steps_per_epoch=num_training_files/batch_size / 5,
        validation_data=validation_generator,
        validation_steps=num_validation_files/batch_size / 5)


def create_pdf(history, model_name):
    print("Generating pdf for " + model_name)

    df = pd.DataFrame(history.history)
    df.index = range(1, len(df)+1)
    df.rename(columns={'accuracy': 'Training Accuracy', 'loss': 'Training Loss',
                       'val_accuracy': 'Validation Accuracy', 'val_loss': 'Validation Loss'}, inplace=True)
    print(df)
    sns.set()
    ax = sns.lineplot(hue='event', marker='o', dashes=False, data=df)
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_xlabel('Epoch')
    ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_ylabel('Value')
    ax.set_title('Model Loss and Accuracy')
    plt.savefig(f'./model_statistics/{model_name}_plot.pdf')

    print("Done")


def generateStatistics(model, test_generator, model_name, num_classes):
    print("Generating Confusion Matrix for " + model_name)
    probabilities = model.predict(test_generator)
    predictions = np.argmax(probabilities, axis=1)

    labels = test_generator.classes
    confusion_matrix = tf.math.confusion_matrix(
        labels, predictions, num_classes=num_classes, weights=None, dtype=tf.dtypes.int32, name=None)

    with open(f'model_statistics/{model_name}_confusion_matrix.txt', 'w') as fh:
        fh.write(str(confusion_matrix))

    with open(f'model_statistics/{model_name}_summary.txt', 'w') as fh:
        model.summary(print_fn=lambda x: fh.write(x + '\n'))

    print("Done")


def testModel(model, batch_size, datasetPath, num_classes, model_name, image_size, preprocess_input, output_statistics=True):
    data_generator = ImageDataGenerator(
        preprocessing_function=preprocess_input)

    print("Testing " + model_name)
    test_generator = data_generator.flow_from_directory(
        f'{datasetPath}/test',
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode='categorical')
    num_files = len(test_generator.filepaths)

    model.evaluate(test_generator, steps=num_files/batch_size)

    if(output_statistics):
        generateStatistics(model, test_generator, model_name, num_classes)
