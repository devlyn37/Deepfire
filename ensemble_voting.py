# Sample program to predict an image using an existing model saved in an .h5 file
from tensorflow import keras
import numpy as np
from tensorflow.keras.applications.resnet50 import preprocess_input as res_preproc
from tensorflow.keras.applications.densenet import preprocess_input as dense_preproc
from tensorflow.keras.applications.mobilenet import preprocess_input as mobile_preproc

from tensorflow.keras.preprocessing.image import ImageDataGenerator

dataset = "/storage/deepfire/subsampledDatasets/forestOnly-1"
image_size = 224
batch_size = 64

# Prepare image preprocessors for each model
res_generator = ImageDataGenerator(preprocessing_function=res_preproc).flow_from_directory(
            f'{dataset}/test',
            target_size=(image_size, image_size),
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=False)

dense_generator = ImageDataGenerator(preprocessing_function=dense_preproc).flow_from_directory(
            f'{dataset}/test',
            target_size=(image_size, image_size),
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=False)

mobile_generator = ImageDataGenerator(preprocessing_function=mobile_preproc).flow_from_directory(
            f'{dataset}/test',
            target_size=(image_size, image_size),
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=False)

res_model = keras.models.load_model("saved_models/resnet50.h5")
dense_model = keras.models.load_model("saved_models/densenet121.h5")
mobile_model = keras.models.load_model("saved_models/mobilenet.h5")

res_prediction = np.array([res.argmax() for res in res_model.predict(res_generator)])
dense_prediction = np.array([res.argmax() for res in dense_model.predict(dense_generator)])
mobile_prediction = np.array([res.argmax() for res in mobile_model.predict(mobile_generator)])


ensemble_prediction = np.rint((res_prediction + dense_prediction + mobile_prediction)/3)


def accuracy(preds):
  return 1-np.count_nonzero(res_generator.labels-preds)/len(res_generator.labels)

print(f'resnet accuracy: {accuracy(res_prediction)}')
print(f'densenet accuracy: {accuracy(dense_prediction)}')
print(f'mobilenet accuracy: {accuracy(mobile_prediction)}')
print(f'ensemble accuracy: {accuracy(ensemble_prediction)}')
