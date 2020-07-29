# Sample program to predict an image using an existing model saved in an .h5 file
from tensorflow import keras
import numpy as np
import ensemble_voting as em
from tensorflow.keras.preprocessing.image import ImageDataGenerator

dataset = "/storage/deepfire/subsampledDatasets/forestOnly-1"
image_size = 224
batch_size = 64

model = em.EnsembleModel(method="pooling")

data_generator = ImageDataGenerator()

test_generator = data_generator.flow_from_directory(
            f'{dataset}/test',
            target_size=(image_size, image_size),
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=False)

prediction = model.evaluate(test_generator)

print(prediction)
