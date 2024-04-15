import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA 
from sklearn.cluster import KMeans
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import jaccard_score
import os
from sklearn.svm import l1_min_c
from sklearn.linear_model import LogisticRegression as LR
from sklearn.model_selection import cross_val_score as cvs
import sys
from sklearn.metrics import roc_auc_score
from sklearn import metrics
from tabulate import tabulate
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate


def create_cross_validation_sets(S, x_train, y_train):
    csets = [] 
    for i in range(len(x_train)): 
        csets.append([x_train[i],y_train[i]])
    np.random.shuffle(csets)
    x_train_r = []
    y_train_r =[]
    for i in range(len(csets)):
        x_train_r.append(csets[i][0])
        y_train_r.append(csets[i][1])
    
    npg = len(x_train)// S
    final = []

    for i in range(S):
        start = i * npg
        end = (i + 1) * npg
        x_sub_train = x_train_r[:start] + x_train_r[end:]
        y_sub_train = y_train_r[:start] + y_train_r[end:]
        x_validation = x_train_r[start:end]
        y_validation = y_train_r[start:end]
        final.append([x_sub_train, y_sub_train, x_validation, y_validation])
    return(final)

#os.chdir(os.pardir)
print(os.getcwd())
prepath = str(os.getcwd())
np.set_printoptions(threshold=sys.maxsize)
results = []
k_mers = ["2","2","4","4","6","6"]
checks = [True,False,True,False,True,False]
for i in range(len(k_mers)):
    k_mer = k_mers[i]
    pca_check = checks[i]

    Windows = True

    if Windows:
        path = "\Dataproject\processed_data\combined_data\split_with_background\\"+k_mer + "mers_extend200\\"
    else:
        path = "/processed_data/combined_data/split_with_background/"+k_mer+"mers_extend200/"

    meth_even = pd.read_csv(prepath+path + "combined_" + k_mer + "mers_meth_even_with_background.tsv", sep="\t")
    meth_odd = pd.read_csv(prepath+path +"combined_"+k_mer+"mers_meth_odd_with_background.tsv", sep="\t")
    unmeth_even = pd.read_csv(prepath+path + "combined_"+k_mer+"mers_unmeth_even_with_background.tsv", sep="\t")
    unmeth_odd = pd.read_csv(prepath+path + "combined_"+k_mer+"mers_unmeth_odd_with_background.tsv", sep="\t")


    meth_even_healthy = meth_even.loc[meth_even["cancer"]=="Healthy"]

    meth_even_unhealthy = meth_even.loc[meth_even["cancer"]!="Healthy"]

    meth_odd_healthy = meth_odd.loc[meth_odd["cancer"]=="Healthy"]

    meth_odd_unhealthy = meth_odd.loc[meth_odd["cancer"]!="Healthy"]

    unmeth_even_healthy = unmeth_even.loc[unmeth_even["cancer"]=="Healthy"]

    unmeth_even_unhealthy = unmeth_even.loc[unmeth_even["cancer"]!="Healthy"]

    unmeth_odd_healthy = unmeth_odd.loc[unmeth_odd["cancer"]=="Healthy"]

    unmeth_odd_unhealthy = unmeth_odd.loc[unmeth_odd["cancer"]!="Healthy"]

    meth_even_healthy = meth_even_healthy.drop(306)

    meth_odd_healthy = meth_odd_healthy.drop(306)

    unmeth_even_healthy = unmeth_even_healthy.drop(306)

    unmeth_odd_healthy = unmeth_odd_healthy.drop(306)




    #index to get get half the healthy 
    tr = np.shape(meth_odd_healthy)[0]//2

    # Training data 

    tr_meth_even_healthy = meth_even_healthy[0:tr]

    tr_meth_odd_healthy = meth_odd_healthy[0:tr]

    tr_unmeth_even_healthy = unmeth_even_healthy[0:tr]

    tr_unmeth_odd_healthy = unmeth_odd_healthy[0:tr]

    # Test data 

    test_meth_even_healthy = meth_even_healthy[tr:]

    test_meth_odd_healthy = meth_odd_healthy[tr:]

    test_unmeth_even_healthy = unmeth_even_healthy[tr:]

    test_unmeth_odd_healthy = unmeth_odd_healthy[tr:]


    # Matrices

    train_matrix = [tr_meth_even_healthy, tr_meth_odd_healthy, tr_unmeth_even_healthy, tr_unmeth_odd_healthy]

    pred_matrix = [test_meth_odd_healthy, test_meth_even_healthy, test_unmeth_even_healthy, test_unmeth_odd_healthy, meth_even_unhealthy, meth_odd_unhealthy, unmeth_even_unhealthy, unmeth_odd_unhealthy]

    train_matrix = pd.concat(train_matrix) 
    train_matrix = train_matrix.iloc[: , :-1]

    pred_matrix = pd.concat(pred_matrix) 
    pred_matrix = pred_matrix.iloc[: , :-1]



    train_targets = [0] * np.shape(tr_meth_even_healthy)[0] + [0] * np.shape(tr_meth_odd_healthy)[0] +  [1] * np.shape(tr_unmeth_even_healthy)[0] + [1] * np.shape(tr_unmeth_odd_healthy)[0] 

    pred_targets = [0] * np.shape(test_meth_even_healthy)[0] + [0] * np.shape(test_meth_odd_healthy)[0] +  [1] * np.shape(test_unmeth_even_healthy)[0] + [1] * np.shape(test_unmeth_odd_healthy)[0] + [0] * np.shape(meth_even_unhealthy)[0] +   [0] * np.shape(meth_odd_unhealthy)[0] +  [1] * np.shape(unmeth_even_unhealthy)[0] +  [1] * np.shape(unmeth_odd_unhealthy)[0]
    print(k_mer," PCA ",pca_check, " created")



    if pca_check == False:
        labels = []
        for i in range(len(meth_even.columns)-1):
            labels.append(meth_even.columns[i])
    else:
        labels = list(range(16))
        #labels = labels[::-1]

    def pca(train_data, test_data): 
        dim_reduction = PCA()
        train_fit = dim_reduction.fit_transform(train_data)
        test_fit = dim_reduction.transform(test_data)
        return [train_fit, test_fit]



    np.logspace(0,0,16)

    labels

    if pca_check == True:
        train_fit, test_fit = pca(train_matrix, pred_matrix)
    else:
        train_fit, test_fit = train_matrix, pred_matrix

    cs = l1_min_c(train_fit, train_targets, loss = 'log') * np.logspace(0,10,32)

    empty_model = LR(penalty = 'l1',solver='liblinear',tol=1e-6,max_iter=1000000,warm_start=True,intercept_scaling=10000.0)
    #empty_model = LogisticRegressionCV(penalty = 'l1',solver='liblinear',tol=1e-6,max_iter=1000,intercept_scaling=10000.0)
    print(k_mer," PCA ",pca_check, "model created")
    coefs_ = []
    previous_score = 0
    all_scores = []
    for c in cs:
        empty_model.set_params(C=c)
        empty_model.fit(train_fit, train_targets)
        preds = empty_model.predict_proba(test_fit)
        score = cvs(empty_model, preds, pred_targets)
        score = score.mean()
        all_scores.append(score)
        print(f'previous score is: {previous_score}')
        print(f'score is: {score}')
        if previous_score > score:
            break
        coefs_.append(empty_model.coef_.ravel().copy())
        previous_score = score
                

    coefs_ = np.array(coefs_)

    #coefs_ = []
    #empty_model.set_params(Cs=cs)
    #empty_model.fit(train_fit, train_targets)
    #coefs_.append(empty_model.coef_.ravel().copy())
    #coefs_ = np.array(coefs_)

    preds = empty_model.predict_proba(test_fit)

    #cv_results = cross_validate(empty_model,train_matrix,pred_matrix)


    score = cvs(empty_model, preds, pred_targets)
    print(score.mean())

    #labels = list(range(16))
    #labels = labels[::-1]


    #colormap = plt.cm.get_cmap('tab20')

    #plt.figure(figsize=(10, 6))

    #if pca_check == True:
        #for i, label in zip(range(len(labels)), labels):
            #plt.plot(np.log10(cs[0:len(coefs_)]), coefs_[:, i], marker="o", label=f"Principal component {label}", color=colormap(i))
    #else:
        #for i, label in zip(range(len(labels)), labels):
            #plt.plot(np.log10(cs[0:len(coefs_)]), coefs_[:, i], marker="o", label=f"label {label}", color=colormap(i))

    #ymin, ymax = plt.ylim()
    #plt.xlabel("log(C)")
    #plt.ylabel("Coefficients")
    #plt.title("Logistic Regression Path")
    #plt.axis("tight")
    #plt.legend()
    #plt.show()


    #all_scores

    #coefs_[-1]

    #cv_results = cross_validate(empty_model,test_fit,pred_targets)

    #cv_results['test_score']

    #print(len(coefs_ [0]))
    print("ROC begin")

    roc_score= roc_auc_score(pred_targets, preds[:, 1])

    fpr, tpr, _ = metrics.roc_curve(pred_targets,  preds[:, 1])

    ROC = [[fpr,tpr],roc_score]
    results.append(ROC)
    #plt.plot(fpr,tpr,label="Lasso & LR, auc="+str(roc_score))
    #plt.legend()
    #plt.xlabel("False positive rate")
    #plt.ylabel("True positive rate")
    print(k_mer," PCA ",pca_check, " done")

table_data = [["2-mer PCA", results[0][1]],
              ["2-mer norm", results[1][1]],
              ["4-mer PCA", results[2][1]],
              ["4-mer norm", results[3][1]],
              ["6-mer PCA", results[4][1]],
              ["6-mer norm", results[5][1]]]
col_names = ["K-mer","ROC score"]
print(tabulate(table_data,headers=col_names))

for i in range(len(results)):
    plt.plot(results[i][0][0],results[i][0][1],label=table_data[i][0] + ", AUC = " + str(table_data[i][1]))
plt.legend()
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.savefig("ROC_curves")

#results = []


