import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA 
from sklearn.cluster import KMeans
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import jaccard_score






meth_even = pd.read_csv("processed_data\combined_data\split_with_background\\2mers_extend200\combined_2mers_meth_even_with_background.tsv", sep="\t")
meth_odd = pd.read_csv("processed_data\combined_data\split_with_background\\2mers_extend200\combined_2mers_meth_odd_with_background.tsv", sep="\t")
unmeth_even = pd.read_csv("processed_data\combined_data\split_with_background\\2mers_extend200\combined_2mers_unmeth_even_with_background.tsv", sep="\t")
unmeth_odd = pd.read_csv("processed_data\combined_data\split_with_background\\2mers_extend200\combined_2mers_unmeth_odd_with_background.tsv", sep="\t")


meth_even = meth_even.loc[meth_even["cancer"]=="Healthy"]
meth_odd = meth_odd.loc[meth_odd["cancer"]=="Healthy"]
unmeth_even = unmeth_even.loc[unmeth_even["cancer"]=="Healthy"]
unmeth_odd = unmeth_odd.loc[unmeth_odd["cancer"]=="Healthy"]


meth_even = meth_even.drop(306)
meth_odd = meth_odd.drop(306)
unmeth_even = unmeth_even.drop(306)
unmeth_odd = unmeth_odd.drop(306)

test = [meth_even, meth_odd, unmeth_even, unmeth_odd]
test = pd.concat(test) 
test = test.iloc[: , :-1]

def pca(train_data, test_data, n_comp): 
    dim_reduction = PCA(n_components= n_comp)
    train_fit = dim_reduction.fit_transform(train_data)
    test_fit = dim_reduction.transform(test_data)
    return [train_fit, test_fit]



training_data = [meth_even, unmeth_even]
training_data = pd.concat(training_data) 
training_data = training_data.iloc[: , :-1]
test_data = [meth_odd, unmeth_odd]
test_data = pd.concat(test_data) 
test_data = test_data.iloc[: , :-1]

train_targets = [0] * 243 + [1] * 243

test_targets = [0] * 243 + [1] * 243

train_fit, test_fit = pca(training_data, test_data, 16)


def pen_lasso(train_fit, train_targets, test_fit): 
    model = Lasso(alpha=0.00001).fit(train_fit, train_targets)
    train_preds = model.predict(train_fit)
    test_preds = model.predict(test_fit)
    train_preds = train_preds.reshape(-1,1)
    test_preds = train_preds.reshape(-1,1)
    return [train_preds, test_preds]

train_preds, test_preds = pen_lasso(train_fit, train_targets, test_fit)

def classification(train_preds, train_targets, test_preds): 
    model = LogisticRegressionCV(random_state=123).fit(train_preds, train_targets)
    LR_test_pred = model.predict(test_preds)
    return LR_test_pred

LR_preds = classification(train_preds, train_targets, test_preds)

LR_preds

#Accuracy 
jaccard_score(test_targets, LR_preds)