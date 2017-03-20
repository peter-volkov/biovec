import argparse
import sys
import os
import gzip
from collections import Counter

os.environ['KERAS_BACKEND'] = 'theano'


import numpy as np
from scipy.spatial.distance import cosine
from Bio import SeqIO

import keras
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Dropout
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score


def get_model(vector_length):
    model = Sequential([        
        Dense(vector_length, activation='relu', name='hidden_1', input_dim=vector_length),
        Dropout(0.25),
        Dense(int(vector_length / 2), activation='relu', name='hidden_2'),
        Dropout(0.25),
        Dense(int(vector_length / 2), activation='relu', name='outer'),
        Dropout(0.25),
        Dense(1, activation='sigmoid', name='protein_family'),
    ])
    model.compile(
        optimizer='adam',
        loss={'protein_family': 'binary_crossentropy'},
        metrics=['accuracy'],
    )
    return model


def fit_model(file_path, epochs):      
    targets = []
    vectors = []
    with open(file_path) as infile:
        for line in infile:
            is_in_family, vector_string = line.rstrip().split('\t', 2)
            targets.append(int(is_in_family))
            vectors.append(np.array(map(float, vector_string.split()), dtype=np.float32))  

    vectors_array = np.array(vectors)
    targets_array = np.array(targets)
    targets_array = targets_array.reshape(-1, 1) 
    vectors, target = None, None
    #print(vectors_array, targets_array)
    #print(targets_array)
    #print(vectors_array.shape, targets_array.shape)
    model = get_model(vector_length=vectors_array.shape[1])

    vectors_train, vectors_test, targets_train, targets_test = train_test_split(vectors_array, 
                                                                                targets_array, 
                                                                                test_size=0.1)
    vectors_array, targets_array = None, None

    history = model.fit(vectors_train, 
                        targets_train, 
                        epochs=epochs, 
                        #validation_split=0.2,
                        validation_data=(vectors_test, targets_test),    
                        batch_size=256, 
                        callbacks=[EarlyStopping(patience=3)])
    score = model.evaluate(vectors_test, targets_test, verbose=1)
    print('Test loss: {}, Test accuracy:', score[0], score[1])
    test_predictions = model.predict(vectors_test, verbose=2, batch_size=256)
    prediction_counter = Counter()

    for index, predicted_float in enumerate(test_predictions):
        predicted = 0 if predicted_float < 0.5 else 1
        correct = targets_test[index][0]
        prediction_counter[predicted==correct] += 1
        #print('{}\t{}\t{}'.format(predicted, correct, predicted==correct))
    sample_file_base_name = os.path.basename(file_path)
    with open('pfam_nn_results/{}.txt'.format(sample_file_base_name), 'w') as outfile:
        result = '{}: t_rate(accuracy)={}'.format(sample_file_base_name, float(prediction_counter[True]) / sum(prediction_counter.values()))
        print(result)    
        outfile.write('{}\n'.format(result))

    with open('pfam_nn_results/{}_model.json'.format(sample_file_base_name), "w") as json_file:
        json_file.write(model.to_json())


def main():      
    parser = argparse.ArgumentParser('Trains NN model over protein vectors')
    parser.add_argument('--sample', type=str, default='training_sample_100.txt')
    parser.add_argument('--epochs', type=int, default=30)
    args = parser.parse_args()

    if os.path.isdir(args.sample):
        sample_directory = args.sample
        for file_name in os.listdir(sample_directory):
            file_path = os.path.join(sample_directory, file_name)
            fit_model(file_path, args.epochs)
    else:
        fit_model(args.sample, args.epochs)
        
    
if __name__ == '__main__':
    main()
