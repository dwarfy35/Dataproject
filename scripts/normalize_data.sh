#!/bin/bash

mkdir -p data_normalized_split

cd ctDNA_frag_meth_split/data/kmer_tsv/2mers_extend200_split/

for file in *.tsv
do
    sum=$(awk -F'\t' '{sum+=$2} END {print sum}' $file)
    awk -v sum=$sum -F'\t' '{print $1"\t"$2/sum}' $file > ../../../../data_normalized_split/2mers_extend200/${file%.tsv}_normalized.tsv
done

cd ../4mers_extend200_split/

for file in *.tsv
do
    sum=$(awk -F'\t' '{sum+=$2} END {print sum}' $file)
    awk -v sum=$sum -F'\t' '{print $1"\t"$2/sum}' $file > ../../../../data_normalized_split/4mers_extend200/${file%.tsv}_normalized.tsv
done

cd ../6mers_extend200_split/

for file in *.tsv
do
    sum=$(awk -F'\t' '{sum+=$2} END {print sum}' $file)
    awk -v sum=$sum -F'\t' '{print $1"\t"$2/sum}' $file > ../../../../data_normalized_split/6mers_extend200/${file%.tsv}_normalized.tsv
done
