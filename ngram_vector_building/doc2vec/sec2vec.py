import gensim 
import sys

if __name__ == '__main__':
    documents = gensim.models.doc2vec.TaggedLineDocument(sys.argv[1])
    model = gensim.models.doc2vec.Doc2Vec(documents, size=250, window=5, min_count=5, workers=32)
    model.wv.save_word2vec_format('doc2vec.txt')
