import lib
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

dataset = '/storage/deepfire/subsampledDatasets/forestOnly-1'
output_pdf = True
output_confusion_matrix = False
model_name = 'simple-neural-net'
batch_size = 64
image_size = 224
num_classes = 2
epochs = 5


def main():
    fire_detector_model = keras.Sequential([
        keras.layers.Flatten(),
        keras.layers.Dense(25, activation="relu"),
        keras.layers.Dense(10, activation="relu"),
        keras.layers.Dense(2, activation='softmax')
    ])

    fire_detector_model.compile(optimizer='rmsprop',
                                loss='categorical_crossentropy',
                                metrics=['accuracy'])

    history = lib.trainModel(dataset, fire_detector_model,
                             epochs, batch_size, image_size, preprocess_input)

    if output_pdf:
        lib.create_pdf(history, model_name)

    lib.testModel(fire_detector_model, batch_size, dataset, num_classes,
                  model_name, image_size, preprocess_input, output_confusion_matrix)
    fire_detector_model.save(f'saved_models/{model_name}.h5')


if __name__ == "__main__":
    main()
