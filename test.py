import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

path = 'ctDNA_frag_meth/data/kmer_tsv/2mers_extend200/'

df1_unmeth = pd.read_csv(path + 'sample1_unmethylated.tsv', sep='\t')

df1_meth = pd.read_csv(path + 'sample1_methylated.tsv', sep='\t')

bgdf1_unmeth= pd.read_csv(path + 'background_unmethylated.tsv', sep='\t')

bgdf1_meth = pd.read_csv(path + 'background_methylated.tsv', sep='\t')

print(df1_unmeth.head())