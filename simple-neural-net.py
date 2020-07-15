import sys
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

matplotlib.use('pdf')

dataset = '../forestOnly-1' # Name of the folder in /storage/deepfire/subsampledDatasets
output_pdf = True
model_name = 'simple-neural-net'

def main():
    '''
    Building Model
    '''
    image_size = 224
    num_classes = 2
    fire_detector_model = keras.Sequential([
        keras.layers.Flatten(),
        keras.layers.Dense(25, activation="relu"),
        keras.layers.Dense(10, activation="relu"),
        keras.layers.Dense(2, activation='softmax')
    ])
    
    fire_detector_model.compile(optimizer='rmsprop',
        loss='categorical_crossentropy',
        metrics=['accuracy'])

    '''
    Training Model
    '''
    from tensorflow.keras.applications.resnet50 import preprocess_input
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    
    data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)
    
    train_generator = data_generator.flow_from_directory(
            f'{dataset}/train',
            target_size=(image_size, image_size),
            batch_size=64,
            class_mode='categorical')

    history = fire_detector_model.fit(
            train_generator,
    	    epochs=5,
            steps_per_epoch=3)

    '''
    Testing Model
    '''
    batch_size = 32
    test_generator = data_generator.flow_from_directory(
            f'{dataset}/test',
            target_size=(image_size, image_size),
            batch_size=batch_size,
            class_mode='categorical')

    num_files = len(test_generator.filepaths)
    fire_detector_model.evaluate(test_generator,
            steps=num_files/batch_size)

    if output_pdf:
        create_pdf(history)

    ''' Create and save Confusion Matrix '''
    probabilities = fire_detector_model.predict(test_generator)
    predicitions = np.argmax(probabilities, axis=1)
    labels = test_generator.classes

    confusion_matrix = tf.math.confusion_matrix(
    labels, predicitions, num_classes=2, weights=None, dtype=tf.dtypes.int32,
    name=None
    )

    with open(f'model_statistics/{model_name}_confusion_matrix.txt','w') as fh:
        # Pass the file handle in as a lambda function to make it callable
        fh.write(str(confusion_matrix))

    ''' Save Model Summary '''
    # Open the file
    with open(f'model_statistics/{model_name}_summary.txt','w') as fh:
        # Pass the file handle in as a lambda function to make it callable
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
