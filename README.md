# OR-GRCNN
The code in this repository is for a binary classification convolutional neural network implemented using the Tensorflow framework for the prediction of OR-ligand binding.

## Requirements and Installation

## How to Run the Model
To generate the data needed for the model, first run the following command:
python entry_point.py --generate_data

Now, you can run the model with:
python entry_point.py --model='gnn' --gnn_cl

To see more about the various flags available, run:
python entry_point.py --help.
