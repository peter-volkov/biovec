import gzip

from Bio import SeqIO


def get_ngrams(protein_strings, ngram_length=3):
    for ngram_offset in xrange(ngram_length):
        for protein_string in protein_strings:
            protein_ngrams = []
            for index in xrange(ngram_offset, len(protein_string) + 1 - ngram_length, ngram_length):
                ngram = protein_string[index:index + ngram_length]
                protein_ngrams.append(ngram)
            yield protein_ngrams


def get_protein_strings():
    with gzip.open('uniprot_sprot.fasta.gz', 'rb') as gzipped_file:
        for record in SeqIO.parse(gzipped_file, "fasta"):
            yield str(record.seq)


def get_all_ngrams():
    protein_strings = get_protein_strings()
    for protein_ngrams in get_ngrams(list(protein_strings), ngram_length=3):
        print(' '.join(protein_ngrams))


if __name__ == '__main__':
    get_all_ngrams()
