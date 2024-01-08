import pandas as pd
import numpy as np
import torch
from joblib import load


def new_Ils_features(cation_smiles_features_dict,anion_smiles_features_dict,tem):
    cation_smiles,anion_smiles,Xs = [],[],[]
    for cat_smiles,cat_feature in cation_smiles_features_dict.items():
        for an_smiles,an_feature in anion_smiles_features_dict.items():
            X = cat_feature + an_feature + [tem]
            cation_smiles.append(cat_smiles)
            anion_smiles.append(an_smiles)
            Xs.append(X)
    return cation_smiles,anion_smiles,Xs

def new_ILs_predicted(cation_smiles_features_dict, anion_smiles_features_dict,tems):
    cons= []
    for t in tems:
        cation_smiles, anion_smiles, Xs = new_Ils_features(cation_smiles_features_dict, anion_smiles_features_dict, t)
        Xs = torch.tensor(np.array(Xs)).to(torch.float32)
        y_con_preds = []
        for i in range(5):
            con_model_path = '../model/ML_model/XGB/model_' + str(i) + '.pkl'
            con_model = load(con_model_path)
            y_con_pred = con_model.predict(Xs)
            y_con_preds.append(y_con_pred)

        con_final = np.mean(y_con_preds, axis=0)
        cons.append(con_final)

    return cation_smiles,anion_smiles,cons

cation_dict_path = '../model/GNN_model/cation_smiles_features_dict.pt'
anion_dict_path = '../model/GNN_model/anion_smiles_features_dict.pt'

cation_smiles_features_dict = torch.load(cation_dict_path)
anion_smiles_features_dict = torch.load(anion_dict_path)

tems = [275, 290, 300, 310, 325]
cation_smiles, anion_smiles, cons = new_ILs_predicted(cation_smiles_features_dict, anion_smiles_features_dict,tems)

con_df = pd.DataFrame(
            {'cation': cation_smiles, 'anion': anion_smiles, 'predicted ionic conductivity(275K)': cons[0],'predicted ionic conductivity(290K)': cons[1],
             'predicted ionic conductivity(300K)': cons[2], 'predicted ionic conductivity(310K)': cons[3], 'predicted ionic conductivity(325K)': cons[4]})


new_ILs_con_path = '../newILs/new_ILs_conductivity.csv'

con_df.to_csv(new_ILs_con_path,index = False)





