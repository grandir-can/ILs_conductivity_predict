import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
import os
inds_path = '../datasets/indx_id/'
data_path = "../datasets/ionic_conductivity.csv"
df = pd.read_csv(data_path)
y = df['ActualValue'].values
y_train_reals,y_test_reals = [], []
for i in range(5):
    train_index = pd.read_csv(os.path.join(inds_path, r'train_ind_' + str(i) + '.csv'))
    test_index = pd.read_csv(os.path.join(inds_path, r'test_ind_' + str(i) + '.csv'))
    train_index = np.array(train_index['train_ind'])
    test_index = np.array(test_index['test_ind'])
    y_train = y[train_index]
    y_test =y[test_index]
    y_train_reals.append(y_train)
    y_test_reals.append(y_test)


config = {
    "font.family":'Arial',
}
rcParams.update(config)

fig, ax = plt.subplots(5,2,figsize=(10,20))
titles = ['a','b','c','d','e']

for i in range(5):
    logbins = np.logspace(np.log10(min(y_train_reals[i])), np.log10(max(y_train_reals[i])), 100)
    sns.distplot(y_train_reals[i], bins=logbins, hist=True, kde=False, color='royalblue', ax=ax[i][0])
    sns.distplot(y_test_reals[i], bins=logbins, hist=True, kde=False, color='orange', ax=ax[i][0])
    ax[i][0].set_xscale('log')
    ax[i][0].set_xlim(10 ** -6, 10 ** 2)
    ax[i][0].set_ylim(0, 300)
    ax[i][0].text(10 ** -5.7 ,271, titles[i]+'1', fontsize=16)
    ax[i][0].set_xlabel("Ionic Conductivity,S/m")
    ax[i][0].set_ylabel("Count")

    sns.kdeplot(y_train_reals[i], log_scale=10, fill=True, color="royalblue", ax=ax[i][1])
    sns.kdeplot(y_test_reals[i], log_scale=10, fill=True, color="orange", ax=ax[i][1])
    ax[i][1].set_xlim(10 ** -6, 10 ** 2)
    ax[i][1].set_ylim(0, 0.7)
    ax[i][1].text(10 ** -5.7, 0.63, titles[i] + '2', fontsize=16)
    ax[i][1].set_xlabel("Ionic Conductivity,S/m")
    ax[i][1].set_ylabel("Probability Conductivity")
ax[0][0].legend(['Training data', 'Test data'], bbox_to_anchor=(0.5,1), frameon=False)
ax[0][1].legend(['Training data', 'Test data'],bbox_to_anchor=(0.5,1), frameon=False)
fig.tight_layout()
plt.savefig("../picture/con_histogram_train_test.tif",format='tif',dpi = 600,bbox_inches = "tight")
plt.show()
