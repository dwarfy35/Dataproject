import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

sample1 = pd.read_csv('data_combined_with_background/combined_2mers_meth_with_background.tsv'
                      , sep='\t', index_col=0)
sample1 = sample1.loc[:, sample1.columns != 'cancer']
sample2 = pd.read_csv('data_combined_with_background/combined_2mers_unmeth_with_background.tsv'
                      , sep='\t', index_col=0)
sample2 = sample2.loc[:, sample2.columns != 'cancer']
sample3 = pd.read_csv('data_combined_with_background/combined_2mers_with_background.tsv'
                      , sep='\t', index_col=0)
sample4 = pd.read_csv('data_combined_with_background/combined_4mers_meth_with_background.tsv', sep='\t', index_col=0)
sample4 = sample4.loc[:, sample4.columns != 'cancer']
sample5 = pd.read_csv('data_combined_with_background/combined_4mers_unmeth_with_background.tsv'
                      , sep='\t', index_col=0)
sample5 = sample5.loc[:, sample5.columns != 'cancer']
sample6 = pd.read_csv('data_combined_with_background/combined_4mers_with_background.tsv'
                      , sep='\t', index_col=0)
sample7 = pd.read_csv('data_combined_with_background/combined_6mers_meth_with_background.tsv'
                      , sep='\t', index_col=0)
sample7 = sample7.loc[:, sample7.columns != 'cancer']
sample8 = pd.read_csv('data_combined_with_background/combined_6mers_unmeth_with_background.tsv'
                      , sep='\t', index_col=0)
sample8 = sample8.loc[:, sample8.columns != 'cancer']
sample9 = pd.read_csv('data_combined_with_background/combined_6mers_with_background.tsv'
                      , sep='\t', index_col=0)


def pca_fit(data):
    pca = PCA(n_components=2)
    pca.fit(data)
    data_pca = pca.transform(data)

    return data_pca


def kmeans_cluster(data, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(data)
    y_kmeans = kmeans.predict(data)
    plt.scatter(data[:, 0], data[:, 1], c=y_kmeans, s=50, cmap='viridis')
    centers = kmeans.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
    plt.show()


pca_fit(sample8)

kmeans_cluster(pca_fit(sample8), 2)
