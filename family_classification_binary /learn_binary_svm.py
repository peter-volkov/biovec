import argparse
import sys
import os
import gzip
from collections import Counter
import cPickle as pickle

import numpy as np
from scipy.spatial.distance import cosine
from Bio import SeqIO

from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, average_precision_score, coverage_error
from sklearn.model_selection import cross_val_score


def get_sample(file_path):
    targets = []
    vectors = []
    with open(file_path) as infile:
        for line in infile:
            is_in_family, vector_string = line.rstrip().split('\t', 2)
            targets.append(int(is_in_family))
            vectors.append(np.array(map(float, vector_string.split()), dtype=np.float32))  

    vectors_array = np.array(vectors)
    targets_array = np.array(targets)
    #targets_array = targets_array.reshape(-1, 1) 
    vectors, target = None, None

    vectors_train, vectors_test, targets_train, targets_test = train_test_split(vectors_array, 
                                                                                targets_array, 
                                                                                test_size=0.1)
    vectors_array, targets_array = None, None
    return vectors_train, vectors_test, targets_train, targets_test


def save_model_metrics(model_params_string, model, vectors_test, targets_test):
    with open('svm_results/{}_results.txt'.format(model_params_string), 'w') as outfile:
        predicted_targets = model.predict(vectors_test)
        outfile.write('score: {}\n'.format(model.score(vectors_test, targets_test)))
        #print('cross_val_test', cross_val_score(model, vectors_test, targets_test, scoring='neg_log_loss'))
        #print('cross_val_train', cross_val_score(model, vectors_train, targets_test, scoring='neg_log_loss'))
        outfile.write('f1_macro: {}\n'.format(f1_score(targets_test, predicted_targets, average='macro')))
        outfile.write('f1_micro: {}\n'.format(f1_score(targets_test, predicted_targets, average='micro')))
        outfile.write('f1_weighted: {}\n'.format(f1_score(targets_test, predicted_targets, average='weighted')))
        outfile.write('accuracy_score: {}\n'.format(accuracy_score(targets_test, predicted_targets)))
        
        prediction_counter = Counter()
        for index, predicted_target in enumerate(predicted_targets): 
            correct_target = targets_test[index]
            if predicted_target and not correct_target:           
                prediction_counter['fp'] += 1
            elif predicted_target and correct_target:           
                prediction_counter['tp'] += 1
            elif not predicted_target and correct_target:           
                prediction_counter['fn'] += 1
            elif not predicted_target and not correct_target:           
                prediction_counter['tn'] += 1

            outfile.write('predicted={} correct={} is_correct={}\n'.format(correct_target, 
                                                                           predicted_target, 
                                                                           predicted_target==correct_target))

        tp_rate = float(prediction_counter['tp']) / (prediction_counter['tp'] + prediction_counter['fn'])
        tn_rate = float(prediction_counter['tn']) / (prediction_counter['tn'] + prediction_counter['fp'])
        t_rate = float(prediction_counter['tn'] + prediction_counter['tp']) / sum(prediction_counter.values())
        result = '{}: tp_rate(specificity) = {} tn_rate(sensitivity) = {} t_rate(accuracy) = {}'.format(model_params_string, tp_rate, tn_rate, t_rate)
        outfile.write('{}\n'.format(result))
        return result
    

def fit_model(file_path, model_type):
    vectors_train, vectors_test, targets_train, targets_test = get_sample(file_path)
    model = None
    if model_type == 'svc_linear':
        model = svm.SVC(kernel='linear') 
    elif model_type == 'svc_rbf':
        model = svm.SVC(kernel='rbf') 
    elif model_type == 'linear_svc':
        model = svm.LinearSVC() 

    model.fit(vectors_train, targets_train)
    model_params_string = '{}_{}'.format(model_type, os.path.basename(file_path))
    with open('pfam_svm_results/{}.pkl'.format(model_params_string), 'wb') as outfile:
        pickle.dump(model, outfile)

    return save_model_metrics(model_params_string, model, vectors_test, targets_test)

#def get_model(file_path):
#    with open(file_path, 'rb') as infile:
#        return pickle.load(infile)

def main():      
    parser = argparse.ArgumentParser('Trains SVM model over protein vectors')
    parser.add_argument('--sample', type=str, default='training_sample_100.txt')
    parser.add_argument('--type', type=str, default='svc_linear')
    args = parser.parse_args()

    if os.path.isdir(args.sample):
        sample_directory = args.sample
        for file_name in os.listdir(sample_directory):
            file_path = os.path.join(sample_directory, file_name)
            print(fit_model(file_path, args.type))
    else:
        fit_model(args.sample, args.type)
        
    
if __name__ == '__main__':
    main()
