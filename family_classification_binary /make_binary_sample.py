import sys
from collections import Counter

import numpy as np


def get_family_distribution(file_path):
    family_distribution = Counter()
    with open(file_path) as infile:
        for line in infile:
            uniprot_id, family, vector_string = line.rstrip().split('\t', 2)
            family_distribution[family] += 1    
    return family_distribution

if __name__ == '__main__':
    
    super_sample_file_path = sys.argv[1]
    #target_family = sys.argv[2] 
    family_distribution = get_family_distribution(super_sample_file_path)
    for target_family, target_family_proteins_count in family_distribution.most_common(10):
        negative_samples = []
        binary_sample_file_path = 'binary_samples_pfam_100/{}_binary_sample.txt'.format(target_family) 
        with open(binary_sample_file_path, 'w') as outfile:                                 
            with open(super_sample_file_path) as infile:
                for line in infile:
                    uniprot_id, family, vector_string = line.rstrip().split('\t', 2)
                    if family == target_family:
                        outfile.write('1\t{}\n'.format(vector_string))
                    else:
                        negative_samples.append(vector_string)

            for negative_sample in np.random.choice(negative_samples, target_family_proteins_count):
                outfile.write('0\t{}\n'.format(negative_sample))