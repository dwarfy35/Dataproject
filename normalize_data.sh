#!/bin/bash

mkdir -p data_normalized

cd ctDNA_frag_meth/data/kmer_tsv/2mers_extend200/

for file in *.tsv
do
    sum=$(awk -F'\t' '{sum+=$2} END {print sum}' $file)
    awk -v sum=$sum -F'\t' '{print $1"\t"$2/sum}' $file > ../../../../data_normalized/2mers_extend200/${file%.tsv}_normalized.tsv
done

cd ../4mers_extend200/

for file in *.tsv
do
    sum=$(awk -F'\t' '{sum+=$2} END {print sum}' $file)
    awk -v sum=$sum -F'\t' '{print $1"\t"$2/sum}' $file > ../../../../data_normalized/4mers_extend200/${file%.tsv}_normalized.tsv
done

cd ../6mers_extend200/

for file in *.tsv
do
    sum=$(awk -F'\t' '{sum+=$2} END {print sum}' $file)
    awk -v sum=$sum -F'\t' '{print $1"\t"$2/sum}' $file > ../../../../data_normalized/6mers_extend200/${file%.tsv}_normalized.tsv
done

cd ../../../../data_normalized/2mers_extend200/
for file in *.tsv
do
    sed -i '1s/^/kmer\tcount\n/' $file
done

cd ../4mers_extend200/
for file in *.tsv
do
    sed -i '1s/^/kmer\tcount\n/' $file
done

cd ../6mers_extend200/
for file in *.tsv
do
    sed -i '1s/^/kmer\tcount\n/' $file
done
