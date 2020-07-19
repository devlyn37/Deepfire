import lib
from tensorflow.keras.layers import maximum
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras import Input, Model, models

dataset = '/storage/deepfire/subsampledDatasets/forestOnly-1'
output_statistics = False
image_size = 224
num_classes = 2
model_name = "ensemble_model"


def main():

    # Grab models
    resnet = models.load_model('./saved_models/resnet50.h5')
    densenet = models.load_model('./saved_models/densenet121.h5')

    # Rename to avoid layer naming conflicts
    resnet._name = "resnet"
    densenet._name = "densenet"

    # Ensemble
    inputs = Input(shape=(image_size, image_size, 3))
    y1 = resnet(inputs)
    y2 = densenet(inputs)
    outputs = maximum([y1, y2])
    ensemble_model = Model(inputs=inputs, outputs=outputs)
    ensemble_model.compile(
        optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

    lib.testModel(ensemble_model, 64, dataset, num_classes,
                  model_name, image_size, preprocess_input, output_statistics)


if __name__ == "__main__":
    main()
