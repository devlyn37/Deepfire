# UAV Forest Fire Detection Model

This repository contains the necessary tools to setup, train, and test a neural network for forest fire recognition.

## Setup

Pull down the model onto the server. Open up `model.py` and make any necessary changes to the global parameters. You can add hidden layers if you want, or just run it as-is. You can also change the data subset used to train / validate.

## Training the Model

Ensure you are using the correct Python3 environment. The server uses `conda` to manage environments, so you will want to run
```
conda activate tensorflow
```
This environment should have all the tools needed to run the model. After that, it's as simple as
```
python3 model.py
```
