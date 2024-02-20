#!/bin/bash
cd $1


# it then goes through all tsv files and replaces the first line with 'kmer' + a tab + 'count'

for file in *.tsv
do
    sed -i '1i kmer\tcount' $file
done
echo "Done"
