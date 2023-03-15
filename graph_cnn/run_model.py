import time
from graph_cnn.model import GraphCNN
import config
import logging as log
import numpy as np
import tensorflow as tf


def convertYClassification(y):
    new_y = np.zeros(len(y), dtype=float)
    for i in range(len(y)):
        new_y[i] = abs(y[i]) >= 1
    print(tf.convert_to_tensor(new_y))
    return tf.convert_to_tensor(new_y)


def trainGNN(
    gnn, 
    X_train, 
    y_train, 
    tr_batch_size=32, 
    tr_optimizer='adagrad', 
    classification=False,
    classification_loss = tf.keras.losses.BinaryCrossentropy,
    classification_weight = 1,
    ):
    if classification:
        model = gnn.classificationModel(hp_optimizer=tr_optimizer, loss=classification_loss)
        y_train = convertYClassification(y_train)
    else:
        model = gnn.createModel(hp_optimizer=tr_optimizer)
    log.info('model fitting started')

    class_weight = {
        0: classification_weight,
        1: 1
    }
    print("CLASS WEIGHT IS ", class_weight[0])
    mod_history = model.fit(
        X_train, y_train, epochs=10, verbose=True,
        batch_size=tr_batch_size,
        class_weight=class_weight,
        validation_split=0.15)
    
    log.info ('model fitting finished successfully')

    return model, mod_history


def testGNN(model, X_test, y_test, classification=False):
    log.info('model evaluation started')
    if classification:
        y_test = convertYClassification(y_test)
    results = model.evaluate(X_test, y_test, verbose=True)
    log.info('The results of the test run are:' + ' '.join([str(r) for r in results]))
    log.info('model evaluation completed')
    return results


def runGNN(model, X_run):
    """Runs the trained model with new data.

    Args:
        model (tf.keras.Model): The pretrained GNN model.
        X_run (List[tf.Tensor]): The 4 tensors corresponding to the input data
    """
    return model.predict(X_run)


def storeResults(results, timing_measures):
    with open('results.txt', 'a') as res_log:
        res_log.write(' '.join([str(r) for r in results]) + ' \n')
        res_log.write('Timing Benchmarks:\n')
        res_log.write(' '.join([str(r) for r in timing_measures]) + '\n')


def runModel(
    batch_size=-1,
    oversamplex=0,
    loss=0,
    weight=1,
    test_frac=0.3,
    classification=False,
    oversampling=0
    ):
    gnn = GraphCNN()
    gnn.initialize()
    
    losses = [tf.keras.losses.BinaryCrossentropy(),
        tf.keras.losses.BinaryFocalCrossentropy(),
        tf.keras.losses.Hinge()]
    loss = losses[loss]
    
    start_train_test_split = time.time()
    X_train_labels, X_test_labels, y_train_labels, y_test_labels = gnn.trainTestSplit(
        model_test_size=test_frac,
        batch_size=batch_size,
        oversampling=oversampling,
        oversamplingx=oversamplex
    )
    end_train_test_split = time.time()

    start_train_data_load = time.time()
    X_train, y_train = gnn.getTensors(X_train_labels, y_train_labels)
    end_train_data_load = time.time()

    start_test_data_load = time.time()
    X_test, y_test = gnn.getTensors(X_test_labels, y_test_labels)
    end_test_data_load = time.time()

    start_model_fitting = time.time()
    model, training_history = trainGNN(
        gnn, X_train, y_train,
        classification=classification,
        classification_loss=loss,
        classification_weight=weight
    )
    end_model_fitting = time.time()

    
    start_model_fitting = time.time()
    results = testGNN(model, X_test, y_test, classification=classification)
    end_model_fitting = time.time()

    timing_measures = [
        end_train_test_split - start_train_test_split,
        end_train_data_load - start_train_data_load,
        end_test_data_load - start_test_data_load,
        end_model_fitting - start_model_fitting,
        end_model_fitting - start_train_test_split
    ]

    storeResults(results, timing_measures)

    
    return model
