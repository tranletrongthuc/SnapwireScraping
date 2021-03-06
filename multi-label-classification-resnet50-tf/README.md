# TensorFlow ResNet50
Custom implementation of ResNet50 Image Classification model using pure TensorFlow 

## Requirements
* Python 3.6 or higher
* Tensorflow 1.x or 2.x

## Usage

### Training
```sh
python train.py -e=[number of epochs] -f=[dataset folder path] -d=[optional: if use TF Debugger]
```

### TensorBoard
To see metrics while training, run tensorboard.  
Plotted metrics are:
- Each batch accuracy, both **train** and **val**
- Each batch loss, both **train** and **val**
- Epoch accuracy, both **train** and **val**
- Epoch loss, both **train** and **val**

```sh
tensorboard --logdir=logs
```

## Project Structure
    .
    ├── data                       
    │   ├── data.py                 # Dataloader  
    │   └── utils.py                # Image Parser
    ├── model                       
    │   ├── resnet.py               # Resnet50 Model
    │   └── layers.py               # Model's Layers 
    ├── logs                        # TensorBoard Logs         
    ├── training                    # Model's Weights
    ├── config.json                 # Configuration File
    └── train.py                    # Training Script
