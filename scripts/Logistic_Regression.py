# %%
import sklearn.decomposition as sk_decomp
from sklearn import linear_model as lm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression as LR
from sklearn.model_selection import cross_val_score as cvs

# %%
meth_even = pd.read_csv("processed_data\combined_data\split_with_background\\2mers_extend200\combined_2mers_meth_even_with_background.tsv", sep="\t")
meth_odd = pd.read_csv("processed_data\combined_data\split_with_background\\2mers_extend200\combined_2mers_meth_odd_with_background.tsv", sep="\t")
unmeth_even = pd.read_csv("processed_data\combined_data\split_with_background\\2mers_extend200\combined_2mers_unmeth_even_with_background.tsv", sep="\t")
unmeth_odd = pd.read_csv("processed_data\combined_data\split_with_background\\2mers_extend200\combined_2mers_unmeth_odd_with_background.tsv", sep="\t")
#unmeth = pd.read_csv("processed_data\combined_data\split_with_background\combined_2mers_unmeth.tsv", sep="\t")

# %%
meth_even

# %%
meth_even = meth_even.loc[meth_even["cancer"]=="Healthy"]
meth_odd = meth_odd.loc[meth_odd["cancer"]=="Healthy"]
unmeth_even = unmeth_even.loc[unmeth_even["cancer"]=="Healthy"]
unmeth_odd = unmeth_odd.loc[unmeth_odd["cancer"]=="Healthy"]

#meth = meth.iloc[: , :-1]
#unmeth = unmeth.iloc[: , :-1]

# %%
nrow = 10

# %%
meth_even_train = meth_even.head(nrow)
meth_odd_train = meth_odd.head(nrow)
unmeth_even_train = unmeth_even.head(nrow)
unmeth_odd_train= unmeth_odd.head(nrow)

# %%
frames = [meth_even,meth_odd,unmeth_even,unmeth_odd]
combo = pd.concat(frames)
combo = combo.iloc[: , :-1]

# %%
pca = sk_decomp.PCA(n_components=16)
combo = pca.fit_transform(combo)

# %%
def column(matrix, i):
    return [row[i] for row in matrix]

# %%
y1 = [0]*nrow
y2 = [0]*nrow
y3 = [1]*nrow
y4= [1]*nrow
Y = [y1,y2,y3,y4]

# %%
y = []
for i in range(4):
    for j in Y[i]:
        y.append(Y[i][j])

# %%
trow = 244

# %%
y1p = [0]*(trow)
y2p = [0]*(trow)
y3p = [1]*(trow)
y4p= [1]*(trow)
Yp = [y1p,y2p,y3p,y4p]

# %%
yp = []
for i in range(4):
    for j in Yp[i]:
        yp.append(Yp[i][j])

# %%
x1 = combo[trow * 0:(trow * 0)+nrow]
x2 = combo[trow * 1:(trow * 1)+nrow]
x3 = combo[trow * 2:(trow * 2)+nrow]
x4 = combo[trow * 3:(trow * 3)+nrow]
X = [x1,x2,x3,x4]

# %%
x = []
for i in range(4):
    for j in range(len(X[i])):
        x.append(X[i][j])

# %%
clf = LR(random_state=0).fit(x,y)

# %%
clf.predict(combo)
#clf.predict_proba(x)
#clf.score(combo,yp)

# %%
scores = cvs(clf, combo, yp, cv=5)
scores

# %%
b = clf.intercept_[0]
w1, w2 = clf.coef_.T
c = -b/w2
m = -w1/w2
xmin, xmax = -1, 2
ymin, ymax = -1, 2.5
xd = np.array([xmin, xmax])
yd = m*xd + c
plt.plot(xd, yd, 'k', lw=1, ls='--')
plt.fill_between(xd, yd, ymin, color='tab:blue', alpha=0.2)
plt.fill_between(xd, yd, ymax, color='tab:orange', alpha=0.2)

plt.scatter(x[trow*0:nrow*1],y[nrow*0:nrow*1], color = "blue")
plt.scatter(x[trow*1:nrow*2],y[nrow*1:nrow*2], color = "green")
plt.scatter(x[trow*2:nrow*3],y[nrow*2:nrow*3], color = "yellow")
plt.scatter(x[nrow*3:nrow*4],y[nrow*3:nrow*4], color = "red")
plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)

# %%
x = column(combo,3)
y = column(combo,0)

# %%
plt.scatter(x[nrow*0:nrow*1],y[nrow*0:nrow*1], color = "blue")
plt.scatter(x[nrow*1:nrow*2],y[nrow*1:nrow*2], color = "green")
plt.scatter(x[nrow*2:nrow*3],y[nrow*2:nrow*3], color = "yellow")
plt.scatter(x[nrow*3:nrow*4],y[nrow*3:nrow*4], color = "red")


# %%
x

# %%
x= np.array(x)
x= x.reshape(-1,1)
y= np.array(y)
y= y.reshape(-1,1)

# %%
clf = lm.Lasso()
clf.fit(x,y)
print(clf.coef_)
print(clf.intercept_)


