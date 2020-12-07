import glob
from Bio import SeqIO
import argparse
import pandas as pd
import random
import numpy as np
import itertools
import gzip

def GetAllKmers(k):
    """Function that calculates all possible combinations of ACGTs of k size."""
    return([''.join(combination) for combination in itertools.product(['A','C','G','T'], repeat=k)])

def GetContig(genome_file, contig_size):
    """Selects randomly a contig of size contig_size in the genome of the genome_file."""
    with gzip.open(genome_file, "rt") as handle:
        parsed_fasta = list(SeqIO.parse(handle, "fasta"))
        genome_size = False
        while genome_size == False:
            random_genome_scaffold = random.sample(parsed_fasta, 1)[0]
            random_genome_sequence = str(random_genome_scaffold.seq)
            if len(random_genome_sequence) > contig_size:
                genome_size = True

        start_index = np.random.randint(0,len(random_genome_sequence)-contig_size)
        return(str(random_genome_sequence[start_index:start_index+contig_size]))

def GetKmerCounts(group, contig, kmers):
    """Counts the occurence of each kmers in a contig."""
    counts_dict = {'Group': group}
    for kmer in kmers:
        counts_dict[kmer] = contig.count(kmer)
    return(counts_dict)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--sample_size', type=int, help="Number of contigs to assess for each category.")
    parser.add_argument('-k', '--k_size',      type=int, help="k size of the kmers.")
    parser.add_argument('-s', '--contig_size', type=int, help="Size of the contigs to assess.")

    args = parser.parse_args()
    sample_size = args.sample_size
    contig_size = args.contig_size
    k = args.k_size

    pro_dirs = ['archaea','bacteria']
    euk_dirs  = ['fungi','protozoa','invertebrate','vertebrate_mammalian','plant','vertebrate_other']

    # flattened list of all genomes in a group
    pro_genomes = [item for sublist in [glob.glob(folder + '/*/*.fna.gz') for folder in pro_dirs] for item in sublist]
    euk_genomes = [item for sublist in [glob.glob(folder + '/*/*.fna.gz') for folder in euk_dirs] for item in sublist]
    print('Prokaryotic genomes:')
    print(pro_genomes)
    print('Eukaryotic genomes:')
    print(euk_genomes)

    kmers = GetAllKmers(k)
    # initiate pandas data frame
    counts_dict = pd.DataFrame(columns=['Group'] + kmers)
    count = 0
    for i in range(sample_size):
        count +=1
        if count > sample_size:
            break
        print('Sample number: ' + str(count), end = '\r')

        # Randomly sample one genome from the files lists
        euk_genome_file = random.sample(euk_genomes, 1)[0]
        pro_genome_file = random.sample(pro_genomes, 1)[0]

        # Get randomly a contig of size contig_size in the genome (using Bio's SeqIO interface to deal with DNA data)
        euk_contig = GetContig(euk_genome_file, contig_size)
        pro_contig = GetContig(pro_genome_file, contig_size)

        # Get the kounts of each kmers in the contigs, create dict to append to the pandas dataframe.
        euk_row = GetKmerCounts('Eukaryote' , euk_contig, kmers)
        pro_row = GetKmerCounts('Prokaryote', pro_contig, kmers)

        # Append dict counts
        counts_dict = counts_dict.append(euk_row, ignore_index=True)
        counts_dict = counts_dict.append(pro_row, ignore_index=True)

    counts_dict.to_csv('../data/Counts_n' + str(sample_size) + '_k' + str(k) + '_s' + str(contig_size) + '.csv', sep=',')

if __name__ == "__main__":
    main()
