import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import rcParams

config = {
    "font.family":'Arial',
    "mathtext.fontset":'stix',
}
rcParams.update(config)
fig, axs = plt.subplots(3, 5,figsize=(15,9),sharex=False, sharey=False)
title = ['GNN+XGBoost', 'GNN+DT', 'GNN+RF', 'GNN+GBRT','GNN+SVM']
error_text = ['88.2%', '94.1%', '85.5%', '83.8%','12.9%']
title_train = ['a', 'b', 'c', 'd','e']
title_test = ['f', 'g', 'h', 'i','j']
title_error = ['k', 'l', 'm', 'n','o']
colors = ['#829ACA',  '#F78769','#61BFA6', '#DD85B8','#F2CC8E']
methods = ['XGB','DT','RF','GBRT','SVR']

for j in range(len(methods)):
    train_df = pd.read_csv('../result/'+methods[j]+'/train_real_pred/train_real_pred_2.csv')
    y_train = train_df['y_train'].values
    y_train_pred = train_df['y_train_pred'].values
    diff_train = (np.array(y_train_pred) - np.array(y_train)) / np.array(y_train) * 100
    diff_train = diff_train.tolist()
    test_df = pd.read_csv('../result/'+methods[j]+'/test_real_pred/test_real_pred_2.csv')
    y_test = test_df['y_test'].values
    y_test_pred = test_df['y_test_pred'].values
    diff_test = (np.array(y_test_pred) - np.array(y_test)) / np.array(y_test) * 100
    diff_test = diff_test.tolist()

    diff_all = diff_train + diff_test


    y_train_short, diff_train_short = [], []
    y_test_short, diff_test_short = [], []
    for k in range(len(diff_train)):
        if -50 < diff_train[k] < 50:
            y_train_short.append(y_train[k])
            diff_train_short.append(diff_train[k])
    for l in range(len(diff_test)):
        if -50 < diff_test[l] < 50:
            y_test_short.append(y_test[l])
            diff_test_short.append(diff_test[l])

    l = 0
    for diff in diff_train_short:
        if -15 <= diff <= 15:
            l = l +1
    rv = round((l /len(diff_train))*100,1)
    print(rv)

    axs[0, j].scatter(y_train, y_train_pred, marker='o', s=12, c=colors[j])
    axs[0, j].legend(['Training set'], frameon=False, fontsize=8)

    axs[0, j].set_title(title[j],fontsize=12,fontweight = 'bold',y=1.05)
    axs[0, j].text(-6, 17, s=title_train[j], fontsize=14,fontweight = 'bold')

    axs[1, j].scatter(y_test,y_test_pred,marker='o',s=12,c='none',edgecolors=colors[j])
    axs[1, j].legend(['Test set'], frameon=False, fontsize=8)
    axs[1, j].text(-6, 17, s=title_test[j], fontsize=14,fontweight = 'bold')


    for i in range(3):
        axs[i, j].spines['bottom'].set_linewidth(1.5);
        axs[i, j].spines['left'].set_linewidth(1.5);
        axs[i, j].spines['top'].set_visible(False)
        axs[i, j].spines['right'].set_visible(False)

    for i in range(2):
        axs[i, j].plot([0, 15], [0, 15], c='#8E8E8E')
        axs[i, j].set_xlim(0, 15)
        axs[i, j].set_ylim(0, 15)
        axs[i, j].set_xticks([0, 5, 10, 15])
        axs[i, j].set_yticks([0, 5, 10, 15])
        axs[i, j].axis('square')
        axs[i, j].set_xlabel("Experimental Ionic Conductivity, S/m", fontsize=10)
        axs[i, j].set_ylabel('Predicted Ionic Conductivity, S/m', fontsize=10)
    print(len(y_train_short))
    print(len(diff_train_short))
    print("------------------------------------------------------------------")
    axs[2, j].scatter(y_train_short, diff_train_short, marker='o', s=8, c=colors[j], alpha=0.8)
    axs[2, j].set_xlabel('Experimental Ionic Conductivity, S/m', fontsize=10)
    axs[2, j].set_ylabel("Relative Deviation (%)", fontsize=10)
    axs[2, j].set_xlim(0, 15)
    axs[2, j].set_ylim(-50, 52)
    axs[2, j].set_xticks([0, 5, 10, 15])
    axs[2, j].set_yticks([-50, -25, 0, 25, 50])
    axs[2, j].plot([0, 15], [15, 15], c='#8E8E8E', linestyle='--')
    axs[2, j].plot([0, 15], [-15, -15], c='#8E8E8E', linestyle='--')
    axs[2, j].text(13, 1, s=str(rv)+'%', fontsize=10)
    axs[2, j].text(-4.8, 52, s=title_error[j], fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('../picture/real_pred_plot.tif', format='tif', dpi = 600)
plt.show()

plt.close()