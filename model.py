import sys
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D

num_classes = 2


'''
Building Model
'''
fire_detector_model = Sequential()

# First section of the NN
resnet_weights_path = 'transferWeights/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
fire_detector_model.add(ResNet50(include_top=False, pooling='avg', weights='imagenet'))
fire_detector_model.layers[0].trainable = False

# Second section of the NN
fire_detector_model.add(Dense(num_classes, activation='softmax'))

# Compile the sections into one NN
fire_detector_model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])


'''
Training Model
'''
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

image_size = 224
data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator = data_generator.flow_from_directory(
        'data/train',
        target_size=(image_size, image_size),
        batch_size=60,
        class_mode='categorical')

validation_generator = data_generator.flow_from_directory(
        'data/validate',
        target_size=(image_size, image_size),
        class_mode='categorical')

history = fire_detector_model.fit(
        train_generator,
        steps_per_epoch=20,
        validation_data=validation_generator,
        validation_steps=3)

'''
Testing Model
'''
batch_size = 32
test_generator = data_generator.flow_from_directory(
        'data/test',
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode='categorical')
num_files = len(test_generator.filepaths)
fire_detector_model.evaluate(test_generator,
        steps=num_files/batch_size)
