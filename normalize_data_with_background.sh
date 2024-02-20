#!/bin/bash

cd data_normalized/2mers_extend200


for sample_file in sample*_methylated_normalized.tsv; do
    sample_number=$(echo "$sample_file" | grep -oP 'sample\d+')

    output_file="../../data_normalized_with_background/2mers_extend200/${sample_number}_methylated_normalized_with_background.tsv"

    awk 'NR==FNR{a[$1]=$2; next} FNR>1{print $1"\t"$2/a[$1]}' "background_methylated_normalized.tsv" "$sample_file" > "$output_file"

    echo "Normalized file for $sample_number generated: $output_file"
done

for sample_file in sample*_unmethylated_normalized.tsv; do
    sample_number=$(echo "$sample_file" | grep -oP 'sample\d+')

    output_file="../../data_normalized_with_background/2mers_extend200/${sample_number}_unmethylated_normalized_with_background.tsv"

    awk 'NR==FNR{a[$1]=$2; next} FNR>1{print $1"\t"$2/a[$1]}' "background_unmethylated_normalized.tsv" "$sample_file" > "$output_file"

    echo "Normalized file for $sample_number generated: $output_file"
done

cd ../4mers_extend200

for sample_file in sample*_methylated_normalized.tsv; do
    sample_number=$(echo "$sample_file" | grep -oP 'sample\d+')

    output_file="../../data_normalized_with_background/4mers_extend200/${sample_number}_methylated_normalized_with_background.tsv"

    awk 'NR==FNR{a[$1]=$2; next} FNR>1{print $1"\t"$2/a[$1]}' "background_methylated_normalized.tsv" "$sample_file" > "$output_file"

    echo "Normalized file for $sample_number generated: $output_file"
done

for sample_file in sample*_unmethylated_normalized.tsv; do
    sample_number=$(echo "$sample_file" | grep -oP 'sample\d+')

    output_file="../../data_normalized_with_background/4mers_extend200/${sample_number}_unmethylated_normalized_with_background.tsv"

    awk 'NR==FNR{a[$1]=$2; next} FNR>1{print $1"\t"$2/a[$1]}' "background_unmethylated_normalized.tsv" "$sample_file" > "$output_file"

    echo "Normalized file for $sample_number generated: $output_file"
done

cd ../6mers_extend200

for sample_file in sample*_methylated_normalized.tsv; do
    sample_number=$(echo "$sample_file" | grep -oP 'sample\d+')

    output_file="../../data_normalized_with_background/6mers_extend200/${sample_number}_methylated_normalized_with_background.tsv"

    awk 'NR==FNR{a[$1]=$2; next} FNR>1{print $1"\t"$2/a[$1]}' "background_methylated_normalized.tsv" "$sample_file" > "$output_file"

    echo "Normalized file for $sample_number generated: $output_file"
done

for sample_file in sample*_unmethylated_normalized.tsv; do
    sample_number=$(echo "$sample_file" | grep -oP 'sample\d+')

    output_file="../../data_normalized_with_background/6mers_extend200/${sample_number}_unmethylated_normalized_with_background.tsv"

    awk 'NR==FNR{a[$1]=$2; next} FNR>1{print $1"\t"$2/a[$1]}' "background_unmethylated_normalized.tsv" "$sample_file" > "$output_file"

    echo "Normalized file for $sample_number generated: $output_file"
done
