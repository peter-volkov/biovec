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


def get_model(vector_length, number_of_families):
    model = Sequential([        
        Dense(vector_length, activation='relu', name='hidden_1', input_dim=vector_length),
        Dropout(0.25),
        Dense(int(vector_length / 2), activation='relu', name='hidden_2'),
        Dropout(0.25),
        Dense(int(vector_length / 2), activation='relu', name='outer'),
        Dropout(0.25),
        Dense(number_of_families, activation='softmax', name='protein_family'),
    ])
    model.compile(
        optimizer='adam',
        loss={'protein_family': 'categorical_crossentropy'},
        metrics=['accuracy'],
    )
    return model


def main():      
    families = []
    vectors = []
    with open(sys.argv[1]) as infile:
        for line in infile:
            uniprot_id, family, vector_string = line.rstrip().split('\t', 2)
            families.append(family)
            vectors.append(np.array(map(float, vector_string.split()), dtype=np.float32))  

    vectors_array = np.array(vectors)
    vectors = None

    label_encoder = preprocessing.LabelEncoder()
    label_encoder.fit(families)
    number_of_classes = len(set(label_encoder.classes_))
    model = get_model(vector_length=vectors_array.shape[1], number_of_families=number_of_classes)

    families_encoded = np.array(label_encoder.transform(families), dtype=np.int32)
    families = None
    families_encoded = families_encoded.reshape((-1, 1))
    families_binary_labels = keras.utils.to_categorical(families_encoded, num_classes=number_of_classes)
    families_encoded = None

    vectors_train, vectors_test, families_train, families_test = train_test_split(vectors_array, 
                                                                                  families_binary_labels, 
                                                                                  test_size=0.1)
    vectors_array, families_binary_labels = None, None

    history = model.fit(vectors_train, 
                        families_train, 
                        epochs=25, 
                        #validation_split=0.2,
                        validation_data=(vectors_test, families_test),    
                        batch_size=256, 
                        callbacks=[EarlyStopping(patience=3)])
    score = model.evaluate(vectors_test, families_test, verbose=1)

    test_predictions = model.predict(vectors_test, verbose=2, batch_size=4096)
    prediction_counter = Counter()
    for index, test_prediction in enumerate(test_predictions):
        predicted_family = label_encoder.inverse_transform(np.argmax(test_prediction))
        actual_family = label_encoder.inverse_transform(np.argmax(families_test[index]))
        prediction_counter[actual_family==predicted_family] += 1
        print('{}\t{}\t{}'.format(actual_family, predicted_family, actual_family==predicted_family))
    print(prediction_counter, float(prediction_counter[True]) / sum(prediction_counter.values()))    
    print(test_predictions)
    print(families_test)
    print('Test loss: {}, Test accuracy:', score[0], score[1])

    with open("nn_256_model.json", "w") as json_file:
        json_file.write(model.to_json())


if __name__ == '__main__':
    main()
