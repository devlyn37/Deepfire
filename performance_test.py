import classify_video
import ensemble_voting as em
import time
import os
from tensorflow import keras
from tensorflow.keras.applications.resnet50 import preprocess_input

INFERENCE_TEST_IMAGE_COUNT = 469

def run_inference_tests(models):
    for model in models:
        print(f'Executing Inference Test for {model}...')
        # get average inference time
        average_inference_time = execute_inference_test(models[model])
        print(f'Inference Test for {model} complete! Average inference time: {average_inference_time}.')

        # save inference time to a file
        with open(f'model_statistics/{model}/inference_time.txt', 'w') as fh:
            fh.write(str(average_inference_time))

    print('Inference Testing Complete.')


def execute_inference_test(model):
    # load images (use aerial_video)
    image_generator = classify_video.get_image_generator('./aerial_video.mp4')

    # total time of all inferences
    total_time = 0

    # perform inferences 
    for image in image_generator:
        start_time = time.time()
        probabilities = model.predict(preprocess_input(classify_video.resize(image)))
        end_time = time.time()
        total_time += (end_time - start_time)
        prediction = classify_video.LABELS[probabilities.argmax()] 

    # return the average inference time over all the classified images
    return total_time/INFERENCE_TEST_IMAGE_COUNT

def init():
    # load models 
    models = {
            'resnet50': keras.models.load_model('saved_models/resnet50.h5'),
            'densenet121': keras.models.load_model('saved_models/densenet121.h5'),
            'mobilenet': keras.models.load_model('saved_models/mobilenet.h5'),
            'ensemble_voting': em.EnsembleModel('voting'),
            'ensemble_pooling': em.EnsembleModel('pooling')
            }

    # ensure statistics directories exist for each model
    for model in models:
        try:
            os.mkdir(f'model_statistics/{model}/')
        except OSError as error:
            pass

    # perform inference tests 
    print('Running Inference Tests...')
    run_inference_tests(models)

init()
