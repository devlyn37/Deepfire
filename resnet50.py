import sys
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
import pandas as pd
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import tensorflow as tf


dataset = '/storage/deepfire/subsampledDatasets/forestOnly-1'
output_pdf = True
hidden_layers = [30]
model_name = 'resnet50'
batch_size = 64

def main():
    '''
    Building Model
    '''
    num_classes = 2
    fire_detector_model = Sequential()
    
    # First section of the NN
    fire_detector_model.add(ResNet50(include_top=False, pooling='avg', weights='imagenet'))
    fire_detector_model.layers[0].trainable = False
    
    # Second section of the NN
    # Add any hidden layers we may want
    for layer in range(len(hidden_layers)):
        fire_detector_model.add(Dense(hidden_layers[layer], activation='relu'))

    # Add our classification layer and display model properties
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
            f'{dataset}/train',
            target_size=(image_size, image_size),
            batch_size=batch_size,
            class_mode='categorical')
    
    validation_generator = data_generator.flow_from_directory(
            f'{dataset}/validate',
            target_size=(image_size, image_size),
            batch_size=64,
            class_mode='categorical')
    
    history = fire_detector_model.fit(
            train_generator,
    	    epochs=5,
            validation_data=validation_generator)
    
    '''
    Testing Model
    '''
    batch_size = 32
    test_generator = data_generator.flow_from_directory(
            f'{dataset}/test',
            target_size=(image_size, image_size),
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=False)
    fire_detector_model.evaluate(test_generator)

    if output_pdf:
        create_pdf(history)

    ''' Create and save confusion matrix '''
    probabilities = fire_detector_model.predict(test_generator)
    predictions = np.argmax(probabilities, axis=1)

    labels = test_generator.classes
    confusion_matrix = tf.math.confusion_matrix(
    	labels, predictions, num_classes=2, weights=None, dtype=tf.dtypes.int32, name=None)

    with open(f'model_statistics/{model_name}_confusion_matrix.txt','w') as fh:
        fh.write(str(confusion_matrix))

    ''' Write Model Summary Statistics '''
    with open(f'model_statistics/{model_name}_summary.txt','w') as fh:
        fire_detector_model.summary(print_fn=lambda x: fh.write(x + '\n'))

    '''
    Save Model
    '''
    fire_detector_model.save(f'saved_models/{model_name}.h5')


def create_pdf(history):
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
    plt.savefig(f'./model_statistics/{model_name}_plot.pdf')


if __name__ == "__main__":
    main()
