# Machine Learning Computer Vision Quickstart

## About

This repository serves as a quickstart base for machine learning computer vision projects in PyTorch.
Features include:
* Pre-defined project structure - simple to extend
* Training loop
* Metric logging with tensorboard
  * Including visualisation figure of results
* Basic starter model
  * Residual UNET with pre-built ResNet blocks
* Save/resume model training
* Fully deterministic model training
* Pre-built Lagrangian optimisation with Modified Differential Method of Multipliers for multi-objective loss functions
* Comprehensive program arguments/flags

## Structure
```
.
├── data: put your datasets here in this manner
│   └── dataset_name
│       └── categorya
│       └── categoryb
├── out: used for storing model and checkpoint data
├── runs: used for storing tensorboard logs
└── src
    ├── datasets: dataloading logic
    ├── evaluation: logic for evaluation and loss calculation
    ├── model: model definition
    │   └── components: reusable model components
    ├── train: training loop
    ├── utils: common utilities
    └── visualiser: logic for creation of graphs and figures
```


## Installation

A virtual environment is recommended for this project. Create and activate a virtual environment as follows
```bash
python3 -m venv venv
source venv/bin/activate
```
Install required packages:
```bash
pip install -r requirements.txt
```

## Running

```bash
python main.py
```

For options run `python src/main.py --help` to show the following:

```
usage: main.py [-h] [-g] [-d] [-t] [-nl] [-e EPOCHS] [-b BATCH_SIZE]
               [-lr LEARNING_RATE] [-s] [-cp CHECKPOINTS] [--resume RESUME]
               [--conditions CONDITIONS]
               dataset

Train and evaluate network

positional arguments:
  dataset               Name of dataset folder within `data` directory

optional arguments:
  -h, --help            show this help message and exit
  -g, --gpu             Evaluate using GPU if available
  -d, --deterministic   Set deterministic/repeatable computation
  -t, --title           Enable title prompt
  -nl, --no-log         Disable tensorboard logging. This can be useful during
                        rapid development
  -e EPOCHS, --epochs EPOCHS
                        Number of epochs to train for
  -b BATCH_SIZE, --batch-size BATCH_SIZE
                        Batch size
  -lr LEARNING_RATE, --learning-rate LEARNING_RATE
                        Learning rate
  -s, --save            Save model_dict upon run completion
  -cp CHECKPOINTS, --checkpoints CHECKPOINTS
                        Epoch interval between checkpoints. No checkpoints
                        otherwise
  --resume RESUME       ID of a previous run to resume running (if specified).
                        Resuming with modified args should be attempted with
                        caution
  --conditions CONDITIONS
                        Path to JSON file containing conditional loss
                        definitions
```

## Checkpoints

Use the above described `--checkpoints` argument, this program will save checkpoints at a user-specified interval of epochs.
Resuming a checkpointed run can be easily done by specifying the `--resume` argument with the respective ID.

Please note that resuming a checkpointed run does not *remember* the previous program arguments and should be called with the same arguments.
Resuming a run with different arguments has been implemented, but should only be attempted with caution and expertise.

When a run that has been checkpointed finishes the final time, it is possible to make tensorboard see the individual checkpointed runs as one continuous run.
To do this, combine the `events.out.tfevents....` under one subdirectory within the `runs` folder.

## Logging

This project uses tensorboard for all logging
```bash
tensorboard --logdir runs --samples_per_plugin images=99999
```
