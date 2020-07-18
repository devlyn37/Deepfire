import lib
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input

dataset = '/storage/deepfire/subsampledDatasets/forestOnly-1'
output_pdf = True
output_confusion_matrix = False
image_size = 224
model_name = 'resnet50'
hidden_layers = [30]
num_classes = 2
batch_size = 64
epochs = 3

# Multiclass Settings
# dataset = '/storage/deepfire/subsampledDatasets/forest-1-smoke-fire-forest'
# num_classes = 3


def main():
    baseModel = ResNet50(include_top=False, pooling='avg', weights='imagenet')
    fire_detector_model = lib.createModel(
        baseModel, hidden_layers, num_classes)

    history = lib.trainModel(dataset, fire_detector_model,
                             epochs, batch_size, image_size, preprocess_input)
    if(output_pdf):
        lib.create_pdf(history, model_name)

    lib.testModel(fire_detector_model, batch_size, dataset,
                  num_classes, model_name, image_size, preprocess_input, output_confusion_matrix)
    fire_detector_model.save(f'saved_models/{model_name}.h5')


if __name__ == "__main__":
    main()
