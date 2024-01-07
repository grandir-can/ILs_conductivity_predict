from rdkit import Chem
import torch
from torch import Tensor
from torch.nn import GRUCell, GRU, Parameter,Linear, ReLU, Sequential
import torch.nn.functional as F
from torch_geometric.nn import GATConv, NNConv,GCNConv, MessagePassing, global_add_pool,Set2Set
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.utils import softmax
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.data import Data, DataLoader
from typing import Optional
import pandas as pd
from joblib import dump,load
from rdkit import DataStructs
from rdkit.Chem import AllChem as Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
import os
import numpy as np

class GATEConv(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int, edge_dim: int,
                 dropout: float = 0.0):
        super().__init__(aggr='add', node_dim=0)

        self.dropout = dropout

        self.att_l = Parameter(torch.Tensor(1, out_channels))
        self.att_r = Parameter(torch.Tensor(1, in_channels))

        self.lin1 = Linear(in_channels + edge_dim, out_channels, False)
        self.lin2 = Linear(out_channels, out_channels, False)

        self.bias = Parameter(torch.Tensor(out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.att_l)
        glorot(self.att_r)
        glorot(self.lin1.weight)
        glorot(self.lin2.weight)
        zeros(self.bias)

    def forward(self, x: Tensor, edge_index: Adj, edge_attr: Tensor) -> Tensor:
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        out += self.bias
        return out

    def message(self, x_j: Tensor, x_i: Tensor, edge_attr: Tensor,
                index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:

        x_j = F.leaky_relu_(self.lin1(torch.cat([x_j, edge_attr], dim=-1)))
        alpha_j = (x_j * self.att_l).sum(dim=-1)
        alpha_i = (x_i * self.att_r).sum(dim=-1)
        alpha = alpha_j + alpha_i
        alpha = F.leaky_relu_(alpha)
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return self.lin2(x_j) * alpha.unsqueeze(-1)


class AtomEmbedding(torch.nn.Module):
    r"""The Attentive FP model for molecular representation learning from the
    `"Pushing the Boundaries of Molecular Representation for Drug Discovery
    with the Graph Attention Mechanism"
    <https://pubs.acs.org/doi/10.1021/acs.jmedchem.9b00959>`_ paper, based on
    graph attention mechanisms.

    Args:
        in_channels (int): Size of each input sample.
        hidden_channels (int): Hidden node feature dimensionality.
        out_channels (int): Size of each output sample.
        edge_dim (int): Edge feature dimensionality.
        num_layers (int): Number of GNN layers.
        num_timesteps (int): Number of iterative refinement steps for global
            readout.
        dropout (float, optional): Dropout probability. (default: :obj:`0.0`)

    """
    def __init__(self, in_channels: int,
                  edge_dim: int, num_layers: int,
                 num_timesteps: int, dropout: float = 0.0):
        super().__init__()

        self.num_layers = num_layers
        self.num_timesteps = num_timesteps
        self.dropout = dropout
        conv = GATEConv(in_channels, in_channels, edge_dim, dropout)
        gru = GRUCell(in_channels, in_channels)
        self.atom_convs = torch.nn.ModuleList([conv])
        self.atom_grus = torch.nn.ModuleList([gru])
        for _ in range(num_layers - 1):
            conv = GATConv(in_channels, in_channels, dropout=dropout,
                           add_self_loops=False, negative_slope=0.01)
            self.atom_convs.append(conv)
            self.atom_grus.append(GRUCell(in_channels, in_channels))

        self.mol_conv = GATConv(in_channels, in_channels,
                                dropout=dropout, add_self_loops=False,
                                negative_slope=0.01)
        self.mol_gru = GRUCell(in_channels, in_channels)

        self.reset_parameters()

    def reset_parameters(self):

        for conv, gru in zip(self.atom_convs, self.atom_grus):
            conv.reset_parameters()
            gru.reset_parameters()
        self.mol_conv.reset_parameters()
        self.mol_gru.reset_parameters()

    def forward(self, x, edge_index, edge_attr, batch):
        """"""
        # Atom Embedding:


        h = F.elu_(self.atom_convs[0](x, edge_index, edge_attr))
        h = F.dropout(h, p=self.dropout, training=self.training)
        x = self.atom_grus[0](h, x).relu_()

        for conv, gru in zip(self.atom_convs[1:], self.atom_grus[1:]):
            h = F.elu_(conv(x, edge_index))
            h = F.dropout(h, p=self.dropout, training=self.training)
            x = gru(h, x).relu_()

        # Molecule Embedding:

        row = torch.arange(batch.size(0), device=batch.device)
        edge_index = torch.stack([row, batch], dim=0)

        out = global_add_pool(x, batch).relu_()
        for t in range(self.num_timesteps):
            h = F.elu_(self.mol_conv((x, out), edge_index))
            h = F.dropout(h, p=self.dropout, training=self.training)
            out = self.mol_gru(h, out).relu_()

        return out

def smiles_to_data(smiles):
    symbols = [
        'B', 'C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br',
        'Co', 'I', 'Sb'
    ]

    data_list = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            print(smi)
            continue
        xs = []
        for atom in mol.GetAtoms():
            symbol = [0.] * len(symbols)
            symbol[symbols.index(atom.GetSymbol())] = 1.
            x = torch.tensor(symbol)
            xs.append(x)
        x = torch.stack(xs, dim=0)

        edge_indices, edge_attrs = [], []
        for bond in mol.GetBonds():
            edge_indices += [[bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]]
            edge_indices += [[bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()]]

            bond_type = bond.GetBondType()
            single = 1. if bond_type == Chem.rdchem.BondType.SINGLE else 0.
            double = 1. if bond_type == Chem.rdchem.BondType.DOUBLE else 0.
            triple = 1. if bond_type == Chem.rdchem.BondType.TRIPLE else 0.
            aromatic = 1. if bond_type == Chem.rdchem.BondType.AROMATIC else 0.
            conjugation = 1. if bond.GetIsConjugated() else 0.
            ring = 1. if bond.IsInRing() else 0.

            edge_attr = torch.tensor([single, double, triple, aromatic, conjugation, ring])

            edge_attrs += [edge_attr, edge_attr]
        if len(edge_attrs) == 0:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0, 6), dtype=torch.float)
        else:
            edge_index = torch.tensor(edge_indices).t().contiguous()
            edge_attr = torch.stack(edge_attrs, dim=0)
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                    smiles=smi)
        data_list.append(data)
    return data_list

def embedding_by_GNN(model,data):
    datasets = DataLoader(data, batch_size=1, shuffle=False)
    smiles,outs = [], []
    for da in datasets:
        smiles.extend(da.smiles)
        out = model(da.x, da.edge_index, da.edge_attr,da.batch)
        out = out.reshape(-1).detach().numpy().tolist()
        outs.append(out)
    smiles_features_dict = dict(zip(smiles, outs))
    return smiles_features_dict

def concat_features_to_ML_input(cation_smiles_features_dict,anion_smiles_features_dict,data_df):
    cation_smiles = data_df['Cation'].values
    anion_smiles = data_df['Anion'].values
    smiles = data_df['Smiles'].values
    smiles = list(set(smiles))
    print("离子液体种类：", len(smiles))
    tem = data_df['Temperature, K'].values

    Xs = []
    for i in range(len(cation_smiles)):
        cation = cation_smiles_features_dict[cation_smiles[i]]
        anion = anion_smiles_features_dict[anion_smiles[i]]
        t = tem[i]
        X = cation + anion + [t]
        Xs.append(X)
    return Xs

if __name__ == '__main__':
    args = {
        'data_load_path': '../datasets/ionic_conductivity.csv',
        'gnn_model':True,
        'gnn_model_path': '../model/GNN_model/',
        'Xs_save_path': '../datasets/'
    }

    df = pd.read_csv(args['data_load_path'])
    cation_smiles = df['Cation'].values
    anion_smiles = df['Anion'].values

    cation_smiles = list(set(cation_smiles))
    anion_smiles = list(set(anion_smiles))

    print('Number of cations:', len(cation_smiles))
    print('Number of anions:', len(anion_smiles))

    cation_data = smiles_to_data(cation_smiles)
    anion_data = smiles_to_data(anion_smiles)

    if args['gnn_model']:
        cation_model = load(os.path.join(args['gnn_model_path'], 'cation_model.pt'))
        anion_model = load(os.path.join(args['gnn_model_path'], 'anion_model.pt'))
    else:
        cation_model = AtomEmbedding(in_channels=12,
                                     edge_dim=6, num_layers=5, num_timesteps=5,
                                     dropout=0.2)
        anion_model = AtomEmbedding(in_channels=12,
                                    edge_dim=6, num_layers=3, num_timesteps=3,
                                    dropout=0.2)

        dump(cation_model, os.path.join(args['gnn_model_path'], 'cation_model.pt'))
        dump(anion_model, os.path.join(args['gnn_model_path'], 'anion_model.pt'))

    cation_smiles_features_dict = embedding_by_GNN(cation_model,cation_data)
    anion_smiles_features_dict = embedding_by_GNN(anion_model,anion_data)
    torch.save(cation_smiles_features_dict, os.path.join(args['gnn_model_path'], 'cation_smiles_features_dict.pt'))
    torch.save(anion_smiles_features_dict, os.path.join(args['gnn_model_path'], 'anion_smiles_features_dict.pt'))



    Xs = concat_features_to_ML_input(cation_smiles_features_dict, anion_smiles_features_dict, df)
    torch.save(Xs, os.path.join(args['Xs_save_path'], 'Xs.pt'))







