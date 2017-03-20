import gzip
from collections import Counter
import sys

from Bio import SeqIO

protein_families = {}
protein_family_stat = Counter()
with gzip.open('Pfam-A.fasta.gz', 'rb') as gzipped_file:
    for record in SeqIO.parse(gzipped_file, "fasta"):  
        # >F0S5U5_PSESL/156-195 F0S5U5.1 PF10417.8;1-cysPrx_C;    
        family_id = record.description.rsplit(';', 2)[-2]
        uniprot_id = record.name.split('/', 1)[0].lstrip('>') 
        protein_families[uniprot_id] = family_id
        protein_family_stat[family_id] += 1

with open('family_stat.txt', 'w') as outfile:
    for family, number_of_proteins in protein_family_stat.iteritems():
        outfile.write('{}\t{}\n'.format(family, number_of_proteins))

min_proteins_in_family = 20
with gzip.open('uniprot_sprot.fasta.gz', 'rb') as gzipped_file, \
     open("uniprot_with_families.fasta", "w") as output_fasta:
    for record in SeqIO.parse(gzipped_file, "fasta"):
        uniprot_id = record.name.split('|')[2] 
        if uniprot_id in protein_families:
            family = protein_families[uniprot_id]
            if protein_family_stat[family] >= min_proteins_in_family:
                record.description += ' PFAM={}'.format(protein_families[uniprot_id])
                SeqIO.write(record, output_fasta, "fasta")
