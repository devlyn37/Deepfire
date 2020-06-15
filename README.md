# UAV Forest Fire Detection Model

This repository contains the necessary tools to setup, train, and test a neural network for forest fire recognition.

## Setup

Pull down the repository onto the server. Once that's done, the only step left is to set up the training, validation, and testing data. This can be configured in `setup.sh`. The data sets are supplied as a space delimited list to each category. Running the setup script will pull the data from the corresponding folder in storage.

## Training the Model

Ensure you are using the correct Python3 environment. The server uses `conda` to manage environments, so you will want to run
```
conda activate tensorflow
```
This environment should have all the tools needed to run the model. After that, it's as simple as
```
python3 model.py
```
