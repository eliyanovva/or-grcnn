"""This module contains some constants used throughout the package, and sets up the logging
    configuration. It also runs a script to ensure that the user is in the correct folder.
"""
import logging
import sys
import os

# Create logging directory
if not os.path.exists('./logs'):
    os.makedirs('./logs')


logging.basicConfig(
    encoding='utf-8',
    level=logging.INFO,
    format='%(asctime)s %(message)s',
    handlers=[
        logging.FileHandler("logs/script.log"),
        logging.StreamHandler(sys.stdout)
    ]
)


PROTEIN_ADJACENCY_MAT_SIZE = 3000
LIGAND_ADJACENCY_MAT_SIZE = 70
PROTEIN_FEATURES_COUNT = 13
LIGAND_FEATURES_COUNT = 12
ATOM_DICT = {'C':0, 'O':1, 'N':2, 'S':3, 'H':4}

#absolute path for a file
MAIN_PACKAGE_DIR = os.path.abspath(os.curdir)
#final directory in filepath
last_dir = os.path.split(os.path.abspath(os.curdir))[-1]
#ensures the final directory is project-protein-fold
while last_dir != "or-grcnn":
    os.chdir("..")
    MAIN_PACKAGE_DIR = os.path.abspath(os.curdir)
    last_dir = os.path.split(MAIN_PACKAGE_DIR)[-1]

#script_dir allows DATA_FILES_PATH to be called from any computer
DATA_FILES_PATH = os.path.join(MAIN_PACKAGE_DIR, 'data_files')
PDB_FILES_PATH = os.path.join(DATA_FILES_PATH, 'pdb_data_files')
BGF_FILES_PATH = os.path.join(DATA_FILES_PATH, 'bgf_files')
MOL_FILES_PATH = os.path.join(DATA_FILES_PATH, 'mol_files')
SMILES_FILES_PATH = os.path.join(DATA_FILES_PATH, 'smiles_files')

MATRIX_DATA_FILES_PATH = os.path.join(MAIN_PACKAGE_DIR, 'graph_cnn', 'data_prep')
MOL_ADJACENCY_PATH = os.path.join(MATRIX_DATA_FILES_PATH, 'mol_adjacency_data')
LIGAND_FEATURE_PATH = os.path.join(MATRIX_DATA_FILES_PATH, 'mol_feature_data')
PROTEIN_ADJACENCY_PATH = os.path.join(MATRIX_DATA_FILES_PATH, 'pdb_adjacency_data')
PROTEIN_FEATURE_PATH = os.path.join(MATRIX_DATA_FILES_PATH, 'pdb_features_data')
MATRIX_FOLDERS = [
    PROTEIN_ADJACENCY_PATH, PROTEIN_FEATURE_PATH, MOL_ADJACENCY_PATH, LIGAND_FEATURE_PATH
]

#TODO: fix constants and logging setup throughout the entire package
PVALUE_THRESHOLD = 0.05

METRIC_ACCURACY = 'precision'
