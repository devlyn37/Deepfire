import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from tensorflow import keras
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

dataset = '/storage/deepfire/subsampledDatasets/forestOnly-1'
image_size = 224
batch_size = 32
method = 'pooling' # must be 'voting' or 'pooling'



def main():
    image_generator = ImageDataGenerator().flow_from_directory(
            f'{dataset}/test',
            target_size=(image_size, image_size),
            class_mode='categorical',
            batch_size=batch_size,
            shuffle=False)
    ensemble = EnsembleModel('voting')
    
    pred = []
    gt = image_generator.labels
    print('performing inference...')
    for x in range(len(image_generator)):
        sample = image_generator.next() # retrieves the next image
        pred = np.concatenate((pred, ensemble.predict(sample[0])))
    
    # accuracy
    acc = 1-(len(np.nonzero(pred-gt))/len(image_generator))
    print(acc)


class EnsembleModel:
    def __init__(self, method):
        self.method = method
        self.models = []
        print('loading resnet')    
        from tensorflow.keras.applications.resnet50 import preprocess_input as res_preproc
        res_model = keras.models.load_model("saved_models/resnet50.h5")
        self.models.append((res_preproc, res_model))
        
        print('loading densenet')
        from tensorflow.keras.applications.densenet import preprocess_input as dense_preproc
        dense_model = keras.models.load_model("saved_models/densenet121.h5")
        self.models.append((dense_preproc, dense_model))
        
        print('loading mobilenet')
        from tensorflow.keras.applications.mobilenet import preprocess_input as mobile_preproc
        mobile_model = keras.models.load_model("saved_models/mobilenet.h5")
        self.models.append((mobile_preproc, mobile_model))

    def evaluate(self, test_generator):
        results = np.array([], dtype=int)
        gt = test_generator.labels
        for x in range(len(test_generator)):
            samples = test_generator.next()
            res = self.predict_from_samples(samples[0])
            results = np.concatenate((results, res))
        accuracy = 1-(len(np.nonzero(results-gt))/len(test_generator))
        return [0, accuracy] # not calculating loss; way too much work for the amount of time left

    def predict(self, test_generator):
        results = np.array([], dtype=int)
        print(len(test_generator))
        for x in range(len(test_generator)):
            samples = test_generator.next()
            res = self.predict_from_samples(samples[0])
            results = np.concatenate((results, res))
        return results

    def predict_from_samples(self, sample):
        predictions = []
        for model in self.models:
            predictions.append(model[1].predict(model[0](sample)))
        if self.method == 'pooling':
            return np.argmax(sum(predictions), axis=1)
        elif self.method == 'voting':
            return np.rint(sum([np.argmax(x, axis=1) for x in predictions])/len(self.models))


if __name__ == "__main__":
    main()
