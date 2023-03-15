"""The logic for the model run is contained here. The input parameters are collected and
the correct model functionality is started.
"""

import os
import shutil
import sys
import logging as log

# Add repo to path!
MAIN_PACKAGE_DIR = os.path.abspath(os.curdir)
sys.path.append(MAIN_PACKAGE_DIR)

import config

from cli_arguments import ModelingParser
from graph_cnn.data_prep import data_generator
from graph_cnn.model import GraphCNN
from graph_cnn.run_model import runModel, runGNN

def createTemporaryDirectories():
    """Generates temporary folders to hold user input.
    """
    os.mkdir(os.path.join(MAIN_PACKAGE_DIR, 'temp_protein_bgf'))
    os.mkdir(os.path.join(MAIN_PACKAGE_DIR, 'temp_ligand_adj_npy'))
    os.mkdir(os.path.join(MAIN_PACKAGE_DIR, 'temp_ligand_feat_npy'))
    os.mkdir(os.path.join(MAIN_PACKAGE_DIR, 'temp_protein_adj_npy'))
    os.mkdir(os.path.join(MAIN_PACKAGE_DIR, 'temp_protein_feat_npy'))


def removeTemporaryDirectories():
    """Removes temporary folders created to hold user input
    """
    try:
        shutil.rmtree(os.path.join(MAIN_PACKAGE_DIR, 'temp_protein_bgf'))
        shutil.rmtree(os.path.join(MAIN_PACKAGE_DIR, 'temp_ligand_adj_npy'))
        shutil.rmtree(os.path.join(MAIN_PACKAGE_DIR, 'temp_ligand_feat_npy'))
        shutil.rmtree(os.path.join(MAIN_PACKAGE_DIR, 'temp_protein_adj_npy'))
        shutil.rmtree(os.path.join(MAIN_PACKAGE_DIR, 'temp_protein_feat_npy'))
    except:
        log.info('Some of the temporary folders do not exist.')


def generateNpyMatrices(protein_path='input_protein_pdb', ligand_path='input_ligand_mol'):
    """Generates the graph representation for proteins and ligands found in the input folders.

    Args:
        protein_path (str): Path to the folder containing the input proteins.
            Defaults to 'input_protein_pdb'.
        ligand_path (str): Path to the folder containing the input ligands.
            Defaults to 'input_ligand_mol'.
    """
    data_generator.generateProteinMatrices(
        pdb_path=protein_path,
        bgf_path='temp_protein_bgf',
        target_adj_path='temp_protein_adj_npy',
        target_feat_path='temp_protein_feat_npy'
    )

    data_generator.generateLigandMatrices(
        mol_path=ligand_path,
        target_adj_path='temp_ligand_adj_npy',
        target_feat_path='temp_ligand_feat_npy'
    )


def generateLabelsList(protein_folder='input_protein_pdb', ligand_folder='input_ligand_mol'):
    """Generates a list of every protein-ligand pair in the input folders.

    Args:
        protein_path (str): Path to the folder containing the input proteins.
            Defaults to 'input_protein_pdb'.
        ligand_path (str): Path to the folder containing the input ligands.
            Defaults to 'input_ligand_mol'.

    Returns:
        list(str): A list with every protein-ligand tuple.
    """
    protein_files = os.listdir(protein_folder)
    mol_files = os.listdir(ligand_folder)
    X_list = []
    for p_file in protein_files:
        if p_file.endswith('.pdb'):
            for m_file in mol_files:
                if m_file.endswith('.mol'):
                    X_list.append([p_file[:-4], m_file[:-4]])

    return X_list


def savePredictions(label_list, results):
    """Saves the results of user input in a new file.

    Args:
        label_list (list): A list of protein-ligand tuples.
        results (list): A list of the predicted binding affinity coefficients for each tuple.
    """
    with open('predicted_results.txt', 'w') as results_file:
        for i in range(len(label_list)):
            results_file.write(
                str(label_list[i][0]) + ',' + str(label_list[i][1]) + ',' + str(results[i]) + '\n'
                )


def ppp():
    """Main logic for the running of the model.
    """
    parser = ModelingParser()
    parser.setup_arguments()
    args = parser.parse_args()

    if  (args.gnn_mode) or (args.model == 'gnn'):
        classification = args.gnn_cl is True
        if args.gnn_mode == 'eval_tuple':
            X = generateLabelsList()
            createTemporaryDirectories()
            log.info('Generated BGF and MOL files in temp directories.')

            try:
                generateNpyMatrices()
                log.info('Generated NPY arrays')

                temp_folders=[
                    'temp_protein_adj_npy',
                    'temp_protein_feat_npy',
                    'temp_ligand_adj_npy',
                    'temp_ligand_feat_npy'
                ]
                g = GraphCNN()
                g.initialize()
                temp_tensors, dummy_y = g.getTensors(X, ['0']*len(X), temp_folders)

                model = runModel(
                    batch_size=args.batch_size,
                    classification=classification,
                    oversampling=args.oversampling)
                predicted_value = runGNN(model, temp_tensors)
                log.info('The predicted binding affinity is ' + str(predicted_value))
                print('The predicted value is ', predicted_value)
            finally:
                removeTemporaryDirectories()

        elif args.gnn_mode == 'eval_protein':

            X = generateLabelsList(ligand_folder=config.MOL_FILES_PATH)
            createTemporaryDirectories()
            try:
                generateNpyMatrices(ligand_path=config.MOL_FILES_PATH)
                log.info('Generated NPY arrays')

                temp_folders=[
                    'temp_protein_adj_npy',
                    'temp_protein_feat_npy',
                    'temp_ligand_adj_npy',
                    'temp_ligand_feat_npy'
                ]
                g = GraphCNN()
                g.initialize()
                temp_tensors, dummy_y = g.getTensors(X, ['0']*len(X), temp_folders)

                model = runModel(
                    batch_size=args.batch_size,
                    classification=classification,
                    oversampling=args.oversampling,
                    oversamplex=args.oversamplex,
                    loss=args.loss,
                    weight=1
                    )
                predicted_values = runGNN(model, temp_tensors)
                log.info('The predicted binding affinity is ' + str(predicted_values))
                print('The predicted value is ', predicted_values)
                savePredictions(X, predicted_values)
            finally:
                removeTemporaryDirectories()
        elif args.gnn_mode == 'eval_ligand':
            pass
        else:
            model = runModel(
                args.batch_size,
                oversamplex=args.oversamplex,
                loss=args.loss, weight=args.class_weight,
                classification=classification,
                oversampling=args.oversampling)



#data_generator.generateLigandMatrices()
#data_generator.generateProteinMatrices()
ppp()
