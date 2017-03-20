#!/usr/bin/env bash

./word2vec_0.1c/word2vec -size 256 -window 25 -cbow 0 \
 -train uniprot_sprot.fasta.ngrams.txt \
 -output uniprot_sprot.fasta.ngram_vectors_256_c.txt \
 -save-vocab uniprot_sprot_fasta_ngram_vocabulary_256_c.txt
