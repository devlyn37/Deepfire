# UAV Forest Fire Detection Model

This repository contains the necessary tools to setup, train, and test a neural network for forest fire recognition.

## Setup

Two models are currently available for testing: ResNet50 and EfficientNetB7. They are run with their respective files using Python3. At the top of the file, you may specify parameters such as hidden layers, and whether or not you want a plot of the training performance over time.

## Training the Model

Ensure you are using the correct Python3 environment. The server uses `conda` to manage environments, so you will want to run
```
conda activate tensorflow
```

If you're using the EfficientNetB7 model, it is not contained in the conda environment. EfficientNet is only support on TensorFlow nightly builds, so this has been installed with Pip. If running on the server, it should already by accessible to you.
