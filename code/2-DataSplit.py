import numpy as np
import pandas as pd
import os
from sklearn.model_selection import StratifiedShuffleSplit

def stratified_shuffle_split_index(index_save_path,X,y):

    sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    i = 0
    for train, test in sss.split(X, y):
        train_ind = pd.DataFrame(train)
        train_ind.columns = ['train_ind']
        train_ind.to_csv(os.path.join(index_save_path, 'train_ind_' + str(i) + '.csv'), index=False)

        test_ind = pd.DataFrame(test)
        test_ind.columns = ['test_ind']
        test_ind.to_csv(os.path.join(index_save_path, 'test_ind_' + str(i) + '.csv'), index=False)
        i += 1

if __name__ == '__main__':
    args = {
        'index_save_path':"../datasets/indx_id/",
        'data_load_path':'../datasets/ionic_conductivity.csv'
    }

    df = pd.read_csv(args['data_load_path'])

    X = df['Smiles'].values
    y = df['ActualValue'].values

    bins = [0, 2, 8, 15]
    y_lable = pd.cut(y, bins, labels=[1, 2, 3])
    stratified_shuffle_split_index(args['index_save_path'], X, y_lable)