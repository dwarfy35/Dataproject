# This script was created by a group member who left. As such our documentation may be somewhat lacking.
# This script calculates and plots clusters for the files in a given directory.

#!/usr/bin/env python3
import os

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

directory_input = input("Enter the directory of the files:")
cluster_input = int(input("How many clusters do you want?"))

all_files = [] # Finds all "tsv" files in directory
for file in os.listdir(directory_input):
    if file.endswith(".tsv"):
        all_files.append(file)


def pca_fit(data):
    pca = PCA(n_components=2)
    pca.fit(data)
    data_pca = pca.transform(data)

    return data_pca


def kmeans_cluster(data, n_clusters): # Calculates k_means clusters and plots them
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(data)
    y_kmeans = kmeans.predict(data)
    plt.scatter(data[:, 0], data[:, 1], c=y_kmeans, s=50, cmap='viridis')
    centers = kmeans.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)


for file in all_files:
    data = pd.read_csv(directory_input + file, sep='\t')
    data = data.loc[:, data.columns != 'cancer']
    pca_fit(data)
    kmeans_cluster(pca_fit(data), cluster_input)
    plt.savefig(file + ".png")
    plt.clf()
