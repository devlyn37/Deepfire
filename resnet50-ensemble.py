import sys
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D, maximum
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Input, Model
import pandas as pd
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
import seaborn as sns

fireDataset = '/storage/deepfire/subsampledDatasets/forestOnly-1'
smokeDataset = '/storage/deepfire/subsampledDatasets/smoke-1'
output_pdf = True
image_size = 224

data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)

def createModel(hidden_layers):
    num_classes = 2
    model = Sequential()

    # First section of the NN
    model.add(ResNet50(include_top=False, pooling='avg', weights='imagenet'))
    model.layers[0].trainable = False

    # Second section of the NN
    # Add any hidden layers we may want
    for layer in range(len(hidden_layers)):
            model.add(Dense(hidden_layers[layer], activation='relu'))

    # Add our classification layer and display model properties
    model.add(Dense(num_classes, activation='softmax'))
    model.summary()

    # Compile the sections into one NN
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def trainModel(dataset, model, epochs, steps_per_epoch, validation_steps, batch_size):
    train_generator = data_generator.flow_from_directory(
            f'{dataset}/train',
            target_size=(image_size, image_size),
            batch_size=batch_size,
            class_mode='categorical')

    validation_generator = data_generator.flow_from_directory(
            f'{dataset}/validate',
            target_size=(image_size, image_size),
    batch_size=batch_size,
            class_mode='categorical')

    return model.fit(
            train_generator,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            validation_data=validation_generator,
            validation_steps=validation_steps)

def create_pdf(history, path):
    df = pd.DataFrame(history.history)
    df.index = range(1, len(df)+1)
    df.rename(columns={'accuracy': 'Training Accuracy', 'loss': 'Training Loss', 'val_accuracy': 'Validation Accuracy', 'val_loss': 'Validation Loss'}, inplace=True)
    print(df)
    sns.set()
    ax = sns.lineplot(hue='event', marker='o', dashes=False, data=df)
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_xlabel('Epoch')
    ax.set_yticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
    ax.set_ylabel('Value')
    ax.set_title('Model Loss and Accuracy')
    plt.savefig(path)

def testModel(model, batch_size, dataset):
    test_generator = data_generator.flow_from_directory(
            f'{dataset}/test',
            target_size=(image_size, image_size),
            batch_size=batch_size,
            class_mode='categorical')
    num_files = len(test_generator.filepaths)
    model.evaluate(test_generator,
            steps=num_files/batch_size)

def main():
    fire_detector_model = createModel([90])
    fire_history = trainModel(fireDataset, fire_detector_model, epochs=5, steps_per_epoch=30, validation_steps=10, batch_size=64)

    # The smoke model has less data to train on than the fire model, values are adjusted as result
    smoke_detector_model = createModel([30])
    smoke_history = trainModel(smokeDataset, smoke_detector_model, epochs=5, steps_per_epoch=20, validation_steps=2, batch_size=64)
    
    # Ensemble
    inputs = Input(shape=(image_size,image_size, 3))
    y1 = fire_detector_model(inputs)
    y2 = smoke_detector_model(inputs)
    outputs = maximum([y1, y2])
    ensemble_model = Model(inputs=inputs, outputs=outputs)
    ensemble_model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

    print("Testing the fire detection model: ")
    testModel(fire_detector_model, 64, fireDataset)
    print("Testing the smoke detection model: ")
    testModel(smoke_detector_model, 64, fireDataset)
    print("Testing the ensemble model: ")
    testModel(ensemble_model, 64, fireDataset)

    if output_pdf:
        create_pdf(smoke_history, "./home/devlyn/smoke_plot.pdf")
        create_pdf(fire_history, "./home/devlyn/fire_plot.pdf")

if __name__ == "__main__":
    main()
