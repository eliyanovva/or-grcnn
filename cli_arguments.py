"""This module contains the Argument Parser object with the expected arguments for modeling.
"""
import argparse


class ModelingParser(argparse.ArgumentParser):
    """This class specifies the Argument Parser which requests for the needed flags
    in modeling the ligand-protein relationship.
    """
    def __init__(self):
        super().__init__(
            description='Predict the binding affinity between your protein and ligand!',
            epilog='Thanks for stopping by!'
        )

    def setup_arguments(self):
        """Adds the expected arguments to the parser class.
        """
        self.add_argument(
            '--oversampling',
            help = 'Determines oversampling strategy.\
                The default value is 0, which means no oversampling.\
                A value of 1 would duplicate all negative samples in the training set x times.',
            type = int,
            default=0
        )

        self.add_argument(
            '--oversamplex',
            help = 'how many times should a minor class value be replicated',
            type=int,
            default = 0
        )

        self.add_argument(
            '--loss',
            help = 'choose BinaryCrossentropy(0), BinaryFocalCrossEntropy(2), Hinge(3)',
            type=int,
            default = 0
        )

        self.add_argument(
            '--class_weight',
            help = 'set the class weight of the minor class',
            type=float,
            default = 0
        )

        self.add_argument(
            '--gnn_cl',
            help = 'Runs the GNN as a binary classificator instead of a regressor.',
            action = 'store_true'
        )

        self.add_argument(
            '--batch_size',
            help = 'Sets the size of the dataset to be used. Defaults to full dataset',
            type = int,
            default = -1
        )

        self.add_argument(
            '--fitting_batch_size',
            help = 'Sets the batch size for model fitting. Defaults to 64.',
            type = int,
            default = 64
        )

        self.add_argument(
            '--optimizer',
            help = 'Sets the optimizer. \
                Choose between "sgd", "adam", "adagrad", and "adamax". Defaults to "adam".',
            default = 'adam'
        )

        self.add_argument(
            '--dropout',
            help = 'Sets the size of the dropout. Defaults to 0.2.',
            type = float,
            default = 0.2
        )

        self.add_argument(
            '--test_train_split',
            help = 'Sets the size of the test train split. Defaults to 0.15.',
            type = float,
            default = 0.3
        )

        self.add_argument(
            '--validation_split',
            help = 'Sets the validation split. Defaults to 0.15.',
            type = float,
            default = 0.15
        )

        self.add_argument(
            '--learning_rate',
            help = 'Sets the learning rate. Defaults to 0.001.',
            type = float,
            default = 0.001
        )

        self.add_argument(
            '--callbacks',
            help = 'Determines whether callbacks will be used. Defaults to True.',
            type = bool,
            default = True
        )

        self.add_argument(
            '--generate_data',
            help = 'The flag generates the matrix data necessary for model training',
            type = bool,
            default = True
        )

        self.add_argument(
            '--model',
            help = 'Specifies the type of model to be used. Choose between "cnn", "gnn", or "rf".\
                If rf_mode or gnn_mode is declared this argument is unnecessary.'
        )

        # run - trains the model from scratch.
        # eval_tuple - runs user input protein and ligand through the model.
        # eval_protein - takes a protein and evaluates its binding affinity with all ligands.
        # eval_ligand - takes a ligand and evaluates its binding affinity with all proteins.
        self.add_argument(
            '--gnn_mode',
            help = 'Choose between "run", "hptuning", "eval_tuple", "eval_ligand", "eval_protein".\
                Defaults to the "run" mode.'
        )

        self.add_argument(
            '--interaction',
            help = "Input interaction coefficient between the protein and ligand inputs."
        )

        self.add_argument(
            '--verbose',
            help = "If used, the program will output and preserve DEBUG level logs."
        )
