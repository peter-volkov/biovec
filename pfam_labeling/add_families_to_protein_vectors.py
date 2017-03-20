import sys

from Bio import SeqIO


def parse_tskv(tskv_string):
    splitted = tskv_string.split('\t')
    return dict(field.split('=', 1) for field in splitted if '=' in field)


def get_uniprot_protein_families():
    protein_families = {}
    for record in SeqIO.parse('uniprot_with_families.min20.fasta', "fasta"): 
        family_id = None
        for element in record.description.split():
            if element.startswith('PFAM'):
                family_id = element.split('=', 1)[1]
        if family_id:
            uniprot_id = record.name.split('|')[-1]
            protein_families[uniprot_id] = family_id
    return protein_families


protein_families = get_uniprot_protein_families()

with open(sys.argv[1]) as protein_vectors:
    for line in protein_vectors:
        uniprot_id, vector = line.rstrip().split('\t', 1)
        if uniprot_id in protein_families:
            print('{}\t{}\t{}'.format(uniprot_id, protein_families[uniprot_id], vector))
