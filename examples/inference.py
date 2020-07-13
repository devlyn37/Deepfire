# Sample program to predict an image using an existing model saved in an .h5 file
from tensorflow import keras
import numpy as np
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

dataset = "../forestOnly-1"
image_size = 224
batch_size = 64

data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)

test_generator = data_generator.flow_from_directory(
            f'{dataset}/test',
            target_size=(image_size, image_size),
            batch_size=batch_size,
            class_mode='categorical')

reconstructed_model = keras.models.load_model("saved_models/resnet50.h5")

prediction = reconstructed_model.predict(test_generator)

print(prediction)
