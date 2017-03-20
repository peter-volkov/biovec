import gensim 
import sys

if __name__ == '__main__':
    min_ngram_frequency = 2
    vector_size = 100
    window_size = 25
    skip_gram = True

    input_file_path = sys.argv[1]
    model = gensim.models.Word2Vec(
        [line.rstrip().split() for line in open(input_file_path)], 
        min_count=min_ngram_frequency, 
        size=vector_size, 
        sg=int(skip_gram), 
        window=window_size)
    model.wv.save_word2vec_format('{}_vectors.txt'.format(input_file_path))

