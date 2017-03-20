import sys
import gzip
from collections import Counter

import numpy as np
from scipy.spatial.distance import cosine
from Bio import SeqIO


def get_ngram_vectors(file_path):
    ngram_vectors = {}
    vector_length = None
    with open(file_path) as infile:
        for line in infile:
            line_parts = line.rstrip().split()   
            # skip first line with metadata in word2vec text file format
            if len(line_parts) > 2:     
                ngram, vector_values = line_parts[0], line_parts[1:]          
                ngram_vectors[ngram] = np.array(map(float, vector_values), dtype=np.float32)
    return ngram_vectors


def normalize(x):
    return x / np.sqrt(np.dot(x, x))


def get_protein_vector(protein_string, ngram_vectors, ngram_length=3):
    vector_length = len(ngram_vectors.values()[0])
    ngrams_sum = np.zeros(vector_length, dtype=np.float32)
    for index in xrange(len(protein_string) + 1 - ngram_length):
        ngram = protein_string[index:index + ngram_length]
        if ngram in ngram_vectors:
            ngram_vector = ngram_vectors[ngram]
            ngrams_sum += ngram_vector
    return normalize(ngrams_sum)


def main():                            
    ngram_vectors_file_path = 'family_classification_sequences.ngrams.vectors.txt'
    if len(sys.argv) > 0:
        ngram_vectors_file_path = sys.argv[1]   
    ngram_vectors = get_ngram_vectors(ngram_vectors_file_path)
    with gzip.open('uniprot_sprot.fasta.gz', 'rb') as gzipped_file:
        for record in SeqIO.parse(gzipped_file, "fasta"):            
            swissprot_long_name = record.name.split('|')[-1] 
            protein_vector = get_protein_vector(str(record.seq), ngram_vectors, ngram_length=3)
            print('{}\t{}'.format(swissprot_long_name, ' '.join(map(str, protein_vector))))


if __name__ == '__main__':
    main()
