import lib
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D

dataset = '/storage/deepfire/subsampledDatasets/forestOnly-1'
output_pdf = True
output_statistics = False
model_name = 'simple-neural-net'
batch_size = 64
image_size = 224
num_classes = 2
epochs = 5

def main():
    fire_detector_model = Sequential([
        Flatten(),
        Dense(25, activation="relu"),
        Dense(10, activation="relu"),
        Dense(2, activation='softmax')
    ])

    fire_detector_model.compile(optimizer='rmsprop',
                                loss='categorical_crossentropy',
                                metrics=lib.METRICS)

    history = lib.trainModel(dataset, fire_detector_model,
                             epochs, batch_size, image_size, preprocess_input)

    if output_pdf:
        lib.create_pdf(history, model_name)

    lib.testModel(fire_detector_model, batch_size, dataset, num_classes,
                  model_name, image_size, preprocess_input, output_statistics)
    fire_detector_model.save(f'saved_models/{model_name}.h5')


if __name__ == "__main__":
    main()
