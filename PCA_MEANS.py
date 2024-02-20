
import sklearn.decomposition as sk
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

fourmer = pd.read_csv("data_combined\combined_4mers.tsv", sep="\t")
pca = sk.PCA (n_components=2)
pca.fit(fourmer)
X1 = pca.transform(fourmer)
kmeans1 = KMeans(n_clusters=2).fit_predict(X1)
incorrect_guesses = 0
correct_guesses = 0
for i in range(len(kmeans1)):
    if i % 2 == 0:
        if kmeans1[i]==0:
            incorrect_guesses += 0
            correct_guesses += 1
        else:
            incorrect_guesses += 1
            correct_guesses += 0
    else:
        if kmeans1[i]==0:
            incorrect_guesses += 1
            correct_guesses += 0
        else: 
            incorrect_guesses+=0
            correct_guesses += 1
res = (correct_guesses/(correct_guesses+incorrect_guesses))
altres = 1-res
fourmer = max(res,altres)

sixmer = pd.read_csv("data_combined\combined_6mers.tsv", sep="\t")
pca = sk.PCA (n_components=2)
pca.fit(sixmer)
X2 = pca.transform(sixmer)
kmeans2 = KMeans(n_clusters=2).fit_predict(X2)
incorrect_guesses = 0
correct_guesses = 0
for i in range(len(kmeans2)):
    if i % 2 == 0:
        if kmeans2[i]==0:
            incorrect_guesses += 0
            correct_guesses += 1
        else:
            incorrect_guesses += 1
            correct_guesses += 0
    else:
        if kmeans2[i]==0:
            incorrect_guesses += 1
            correct_guesses += 0
        else: 
            incorrect_guesses+=0
            correct_guesses += 1
res = (correct_guesses/(correct_guesses+incorrect_guesses))
altres = 1-res
sixmer = max(res,altres)

twomer = pd.read_csv("data_combined\combined_2mers.tsv", sep="\t")
pca = sk.PCA (n_components=2)
pca.fit(twomer)
X = pca.transform(twomer)
kmeans = KMeans(n_clusters=2).fit_predict(X)
incorrect_guesses = 0
correct_guesses = 0
for i in range(len(kmeans)):
    if i % 2 == 0:
        if kmeans[i]==0:
            incorrect_guesses += 0
            correct_guesses += 1
        else:
            incorrect_guesses += 1
            correct_guesses += 0
    else:
        if kmeans[i]==0:
            incorrect_guesses += 1
            correct_guesses += 0
        else: 
            incorrect_guesses+=0
            correct_guesses += 1
res = (correct_guesses/(correct_guesses+incorrect_guesses))
altres = 1-res
twomer = max(res,altres)

print("2-mer accuracy",twomer)
print("4-mer accuracy", fourmer)
print("6-mer accuracy", sixmer)

print("2-mer preds")
print(kmeans)
print("4-mer preds")
print(kmeans1)
print("6-mers preds")
print(kmeans2)

