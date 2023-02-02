import torch
import torch.nn as nn
import torch.sparse as sparse
import torch.nn.functional as F
from torch.nn import Parameter, BCEWithLogitsLoss, Linear, Sequential, BatchNorm1d, ReLU, Tanh, Dropout, ModuleList, Identity, LeakyReLU, Softmax
from torch_geometric.nn import GCNConv, GENConv, SAGEConv, GINEConv, GATConv, GATv2Conv, LayerNorm
from torch_geometric.nn import global_mean_pool, GlobalAttention
from torch_geometric.nn.pool.topk_pool import topk, filter_adj

import numpy as np
import math

from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, average_precision_score, roc_curve, PrecisionRecallDisplay
from lifelines.utils import concordance_index
from sklearn.model_selection import StratifiedKFold
from collections import defaultdict
import random

from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

import argparse
import os
import torch.optim as optim

import pickle
import pandas as pd
    
SEED = 0

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

class Sampler(object):
    """Base class for all Samplers.
    Every Sampler subclass has to provide an __iter__ method, providing a way
    to iterate over indices of dataset elements, and a __len__ method that
    returns the length of the returned iterators.
    """

    def __init__(self, data_source):
        pass

    def __iter__(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class StratifiedSampler(Sampler):
    """Stratified Sampling
    Provides equal representation of target classes in each batch
    """
    def __init__(self, class_vector, batch_size) :
        """
        Arguments
        ---------
        class_vector : torch tensor
            a vector of class labels
        batch_size : integer
            batch_size
        """
        self.n_splits = int(class_vector.size(0) / batch_size)
        self.class_vector = class_vector

    def gen_sample_array(self) :
        
        try : from sklearn.model_selection import StratifiedShuffleSplit
        except : print('Need scikit-learn for this functionality')
        import numpy as np
        
        s = StratifiedShuffleSplit(n_splits=self.n_splits, test_size=0.5)
        X = torch.randn(self.class_vector.size(0),2).numpy()
        y = self.class_vector.numpy()
        s.get_n_splits(X, y)

        train_index, test_index = next(s.split(X, y))
        return np.hstack([train_index, test_index])

    def __iter__(self):
        return iter(self.gen_sample_array())

    def __len__(self):
        return len(self.class_vector)

    
class ConvBlock_v1(torch.nn.Module) :
    
    def __init__(self, input_dim, output_dim, dropout, norm=LayerNorm, last_act=ReLU, residual=False) :
        
        super().__init__()
        self.conv = GCNConv(input_dim, output_dim)
        self.norm = norm(output_dim)
        self.do = Dropout(dropout)
        if last_act : self.act = Identity()
        else : self.act = last_act()
        self.residual = residual
        
    def forward(self, x, edge_index) : 
        out = self.conv(x, edge_index)
        out = self.norm(out)
        out = self.act(out)
        out = self.do(out)
        if self.residual : return x + out
        else : return out
        
        
class ConvBlock_v2(torch.nn.Module) :
    
    def __init__(self, 
                 input_dim, output_dim, heads, dropout, 
                 norm=LayerNorm, last_act=ReLU, residual=False) :
        
        super().__init__()
        self.conv = GATConv(input_dim, output_dim, heads=heads)
        self.norm = norm(output_dim)
        self.do = Dropout(dropout)
        if last_act : self.act = Identity()
        else : self.act = last_act()
        self.residual = residual
        
    def forward(self, x, edge_index, return_attention_weights=False) : 
        if return_attention_weights :
            out, att = self.conv(x, edge_index, return_attention_weights=True)
            out = self.norm(out)
            out = self.act(out)
            out = self.do(out)
            if self.residual : return x + out, att
            else : return out, att
        else :
            out = self.conv(x, edge_index)
            out = self.norm(out)
            out = self.act(out)
            out = self.do(out)
            if self.residual : return x + out
            else : return out
        
        
class Source2Token(nn.Module) :

    def __init__(self, d_h, dropout=0.2) :
        super(Source2Token, self).__init__()

        self.d_h = d_h
        self.latent_dim = self.d_h * 4
        
        self.fc1 = nn.Linear(d_h, self.latent_dim)
        self.fc2 = nn.Linear(self.latent_dim, 1)
        
        self.norm = LayerNorm(self.latent_dim)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.Tanh()
        
    def forward(self, x, return_attention_weights=False) :
        out = self.act(self.norm(self.fc1(x)))
        out = self.fc2(out).squeeze(-1)
        
        att = F.softmax(out, dim=1)
        out = torch.sum(att.unsqueeze(2) * x, dim=1)

        if return_attention_weights :
            return out, att
        else : 
            return out


class InterPropagation_Source2Token(nn.Module) : # self-attention
    
    def __init__(self, d_h, dropout=0.1) :
        super(InterPropagation_Source2Token, self).__init__()
        self.d_h = d_h
        self.att_layer = Source2Token(d_h)
    
    def forward(self, x, indicator, return_attention_weights=False, level3=False) :
        if return_attention_weights :
            out_list, att_list = [], []
            for rep_mask in indicator :
                out, att = self.att_layer(x[:, rep_mask == 1, :], return_attention_weights=True)
                out_list.append(out)
                att_list.append(att)
            return torch.stack(out_list, dim=1), att_list
        else :
            return torch.stack([self.att_layer(x[:, rep_mask == 1, :]) for rep_mask in indicator], dim=1)
    
    
class NegativeLogLikelihood(nn.Module) :
    
    def __init__(self) :
        super(NegativeLogLikelihood, self).__init__()

    def forward(self, risk_pred, y, e, DEVICE) :
        mask = torch.ones(y.shape[0], y.shape[0]).to(DEVICE)
        mask[(y.T - y) > 0] = 0
        log_loss = torch.exp(risk_pred) * mask
        log_loss = torch.sum(log_loss, dim=0) / torch.sum(mask, dim=0)
        log_loss = torch.log(log_loss).reshape(-1, 1)
        neg_log_loss = -torch.sum((risk_pred-log_loss) * e) / torch.sum(e)
        return neg_log_loss
    
    
class Regularization(nn.Module) :
    
    def __init__(self, order, weight_decay):
        super(Regularization, self).__init__()
        self.order = order
        self.weight_decay = weight_decay

    def __call__(self, model):
        reg_loss = 0
        for name, w in model.named_parameters():
            if 'weight' in name:
                reg_loss = reg_loss + torch.norm(w, p=self.order)
        reg_loss = self.weight_decay * reg_loss
        return reg_loss
    
    
def calculate_c_index(risk_pred, y, e) :
    
    if not isinstance(y, np.ndarray) : y = y.detach().cpu().numpy()
    if not isinstance(risk_pred, np.ndarray) : risk_pred = risk_pred.detach().cpu().numpy()
    if not isinstance(e, np.ndarray) : e = e.detach().cpu().numpy()
        
    return concordance_index(y, risk_pred, e)



class HierarchicalGNN(torch.nn.Module) :
    
    def __init__(self, 
                 level_1_nodes, level_2_nodes, level_3_nodes,
                 level_1_edges, level_2_edges,
                 level_11_edge_index, level_22_edge_index,
                 level_21_indicator, level_32_indicator, level_13_indicator,
                 input_dim, hidden_dim, batch_num, DEVICE,
                 level_3_strategy="concat", agg_strategy="concat",
                 gnn_dropout=0.2, attributes=None) :
        
        super(HierarchicalGNN, self).__init__()
        
        self.attributes = attributes
        self.attribute_dim = 0
        if not self.attributes is None : self.attribute_dim = self.attributes.shape[1]            
            
        self.level_3_strategy = level_3_strategy
        self.agg_strategy = agg_strategy
        
        self.level_1_nodes = level_1_nodes
        self.level_2_nodes = level_2_nodes
        self.level_3_nodes = level_3_nodes
        
        self.level_1_edges = level_1_edges
        self.level_2_edges = level_2_edges
        self.level_11_edge_index = level_11_edge_index
        self.level_22_edge_index = level_22_edge_index
        
        self.batch_lv1_edge_index = torch.cat([self.level_11_edge_index + (i * self.level_1_nodes) for i in range(batch_num)], dim=1).to(DEVICE)
        self.batch_lv2_edge_index = torch.cat([self.level_22_edge_index + (i * self.level_2_nodes) for i in range(batch_num)], dim=1).to(DEVICE)
        
        self.level_21_indicator = level_21_indicator.to(DEVICE) # two-dim list
        self.level_32_indicator = level_32_indicator.to(DEVICE) # two-dim list
        self.level_13_indicator = level_13_indicator # just a list of indices
        
        self.input_dim = 1 + self.attribute_dim
        self.hidden_dim = hidden_dim
        
        self.level_11_propagation_1 = ConvBlock_v1(self.input_dim, self.hidden_dim, dropout=gnn_dropout, residual=False)
        self.level_11_propagation_2 = ConvBlock_v1(self.hidden_dim, self.hidden_dim, dropout=gnn_dropout, residual=True, last_act=True)
        self.level_12_propagation = InterPropagation_Source2Token(self.hidden_dim)
        self.level_22_propagation_1 = ConvBlock_v2(self.hidden_dim, self.hidden_dim, heads=1, dropout=gnn_dropout, residual=False, last_act=True)
        self.level_23_propagation = InterPropagation_Source2Token(self.hidden_dim)
        
        self.device = DEVICE
        
        if self.level_3_strategy == "concat" : self.output_dim = 2 * (self.hidden_dim)
        else : self.output_dim = self.hidden_dim
        if self.agg_strategy == "concat" : self.output_dim = self.level_3_nodes * self.output_dim
        
    def forward(self, x, batch_num, return_attention_weights=False) :
        
        lv1_edge_index = self.batch_lv1_edge_index[:, : batch_num * self.level_1_edges]
        lv2_edge_index = self.batch_lv2_edge_index[:, : batch_num * self.level_2_edges]
                
        if x.ndim == 2 : x = x.unsqueeze(2)
        
        #lv1_x = x
        #lv2_x = torch.stack([torch.mean(x[:, rep_mask == 1, :].squeeze(), dim=1) for rep_mask in self.level_21_indicator], dim=1).unsqueeze(2)
        
        if not self.attributes is None :
            lv1_attributes = torch.cat([self.attributes for i in range(batch_num)], dim=0).view(batch_num, self.attributes.shape[0], self.attributes.shape[1]).float().to(self.device)
            x = torch.cat([x, lv1_attributes], dim=2)
        
        lv1 = x
        lv1 = lv1.view(-1, self.input_dim)
        lv1 = self.level_11_propagation_1(lv1, lv1_edge_index)
        lv1 = self.level_11_propagation_2(lv1, lv1_edge_index)
        lv1 = lv1.view(batch_num, self.level_1_nodes, self.hidden_dim)
        #lv1 = torch.cat((lv1_x, lv1), dim=2)
        
        if return_attention_weights : 
            lv2, lv_12_att = self.level_12_propagation(lv1, self.level_21_indicator, return_attention_weights=True)
            lv2 = lv2.view(-1, self.hidden_dim)
            lv2, lv_22_att = self.level_22_propagation_1(lv2, lv2_edge_index, return_attention_weights=True)
            lv2 = lv2.view(batch_num, self.level_2_nodes, self.hidden_dim)
            #lv2 = torch.cat((lv2_x, lv2), dim=2)
            lv3, lv_23_att = self.level_23_propagation(lv2, self.level_32_indicator, return_attention_weights=True, level3=True)
        else :
            lv2 = self.level_12_propagation(lv1, self.level_21_indicator)
            lv2 = lv2.view(-1, self.hidden_dim)
            lv2 = self.level_22_propagation_1(lv2, lv2_edge_index)
            lv2 = lv2.view(batch_num, self.level_2_nodes, self.hidden_dim)
            #lv2 = torch.cat((lv2_x, lv2), dim=2)
            lv3 = self.level_23_propagation(lv2, self.level_32_indicator)
     
        if self.level_3_strategy == "add" :
            lv3 = lv1[:, self.level_13_indicator, :] + lv3
            if self.agg_strategy == "concat" : p = lv3.view(batch_num, -1)
        elif self.level_3_strategy == "mean" :
            lv3 = (lv1[:, self.level_13_indicator, :] + lv3) / 2
            if self.agg_strategy == "concat" : p = lv3.view(batch_num, -1)
        elif self.level_3_strategy == "concat" : 
            lv3 = torch.cat((lv1[:, self.level_13_indicator, :], lv3), dim=2)
            if self.agg_strategy == "concat" : p = lv3.view(batch_num, -1)
        elif self.level_3_strategy == "attention" :
            if self.agg_strategy == "concat" : p = lv3.view(batch_num, -1)
                
        if self.agg_strategy == "mean" : p = torch.mean(lv3, dim=1)
        elif self.agg_strategy == "sum" : p = torch.sum(lv3, dim=1)
        elif self.agg_strategy == "weighted" : p = self.aggregation(p)
        
        if return_attention_weights :
            return p, lv_12_att, lv_22_att, lv_23_att
        else :
            return p
        
        
class MLP(nn.Module) :
    
    def __init__(self, dims, dropout, norm, activation) :
        super(MLP, self).__init__()
        self.drop = dropout
        self.norm = norm
        self.dims = dims
        self.activation = activation
        self.model = self._build_network()

    def _build_network(self) :
        layers = []
        for i in range(len(self.dims) - 1) :
            if i != 0 and self.drop is not None : 
                layers.append(nn.Dropout(self.drop))
            layers.append(nn.Linear(self.dims[i], self.dims[i+1]))
            if i+1 != len(self.dims) - 1 :
                if self.norm : layers.append(nn.BatchNorm1d(self.dims[i+1]))
                layers.append(eval('nn.{}()'.format(self.activation)))
        return nn.Sequential(*layers)

    def forward(self, X) :
        return self.model(X)
    
    
class MLKG_SurvPredictor(torch.nn.Module) :
    
    def __init__(self, 
                 level_1_nodes, level_2_nodes, level_3_nodes,
                 level_1_edges, level_2_edges,
                 level_11_edge_index, level_22_edge_index,
                 level_21_indicator, level_32_indicator, level_13_indicator,
                 input_dim, hidden_dim, batch_num, device,
                 level_3_strategy="concat", agg_strategy="concat",
                 gnn_dropout=0.2, attributes=None,
                 surv_dropout=0.2, surv_norm=True, surv_activation="ReLU") :
        
        super(MLKG_SurvPredictor, self).__init__()
        
        self.GNN = HierarchicalGNN(level_1_nodes, level_2_nodes, level_3_nodes,
                                   level_1_edges, level_2_edges,
                                   level_11_edge_index, level_22_edge_index,
                                   level_21_indicator, level_32_indicator, level_13_indicator,
                                   input_dim, hidden_dim, batch_num, device,
                                   level_3_strategy, agg_strategy,
                                   gnn_dropout, attributes)
        
        #surv_dims = [self.GNN.output_dim, self.GNN.output_dim * 2, self.GNN.output_dim * 2, 1]
        surv_dims = [self.GNN.output_dim, 128, 128, 1]
        self.MLP = MLP(surv_dims, surv_dropout, surv_norm, surv_activation)
        
    def forward(self, x, batch_num, return_attention_weights=False) :
        
        if return_attention_weights :
            x, lv_12_att, lv_22_att, lv_23_att = self.GNN(x, batch_num, return_attention_weights=True)
            x = self.MLP(x)
            return x, lv_12_att, lv_22_att, lv_23_att
        
        else :
            x = self.GNN(x, batch_num)
            x = self.MLP(x)
            return x


class ExpressionDataset(Dataset) :
    
    def __init__(self, X, e, y) :
        self.X, self.e, self.y = X, e, y
        self.num_features = X.shape[1]

    def __getitem__(self, item):
        X_item = self.X[item]
        e_item = self.e[item]
        y_item = self.y[item]
        
        X_tensor = torch.from_numpy(X_item).float()
        e_tensor = torch.from_numpy(e_item).float()
        y_tensor = torch.from_numpy(y_item).float()
        return X_tensor, y_tensor, e_tensor

    def __len__(self):
        return self.X.shape[0]
    

def argument_parsing() :

    parser = argparse.ArgumentParser(description='Multi-level Knowledge Graph Model')

    parser.add_argument('--SOURCE_GENE_SET', type=str, help='source gene set name', default='CGC_Hallmark_BreastCancer')
    parser.add_argument('--TARGET_GENE_SET', type=str, help='target gene set name', default='UpdatedOncotypeDXCancer')
    
    parser.add_argument('--BLOCK_CUTOFF', type=int, help='cascade block cutoff', default=2)
    parser.add_argument('--BRIDGE_CUTOFF', type=int, help='bridge cutoff', default=1)
    parser.add_argument('--EVIDENCE_CUTOFF', type=int, help='TFTG evidence cutoff', default=5)
    
    parser.add_argument('--LENGTH_CUTOFF', type=int, help='block length cutoff', default=2)
    
    parser.add_argument('--EPOCHS', type=int, help='training epochs', default=200)
    parser.add_argument('--BATCH_SIZE', type=str, help='batch size', default=512)
    parser.add_argument('--DEVICE', type=str, help='device', default="cuda:0")
    parser.add_argument('--HIDDEN_DIM', type=int, help='hidden dimension', default=16)
    parser.add_argument('--LEARNING_RATE', type=float, help='learning rate', default=1e-4)
    parser.add_argument('--PATIENCE', type=int, help='patience epochs for early stopping', default=10)

    args = parser.parse_args()
    
    return args


def gene_one_hot_encoding(hallmark_gene_sets, hallmark_names, target_genes) :
    hallmark_by_gene = np.zeros((len(hallmark_gene_sets), len(target_genes)))
    for idx1, hallmark in enumerate(hallmark_gene_sets) :
        for idx2, gene in enumerate(target_genes) :
            if gene in hallmark :
                hallmark_by_gene[idx1, idx2] = 1
    return pd.DataFrame(data=hallmark_by_gene, index=hallmark_names, columns=target_genes)


def train(train_dataset, val_dataset, test_dataset, device) : 
    
    model = MLKG_SurvPredictor(level_1_nodes=level_1_node_number, level_2_nodes=level_2_node_number, level_3_nodes=level_3_node_number,
                               level_1_edges=level_11_G_torch.edge_index.shape[1], level_2_edges=level_22_G_torch.edge_index.shape[1],
                               level_11_edge_index=level_11_G_torch.edge_index, level_22_edge_index=level_22_G_torch.edge_index,
                               level_21_indicator=level_21_indicator, level_32_indicator=level_32_indicator, level_13_indicator=level_13_indicator,
                               input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, batch_num=BATCH_NUM, device=device,
                               level_3_strategy=LEVEL_3_AGG, agg_strategy=AGG_STRATEGY,
                               surv_dropout=SURV_DROPOUT, gnn_dropout=GNN_DROPOUT).to(device)
    
    criterion = NegativeLogLikelihood().to(device)
    reg_criterion = Regularization(order=2, weight_decay=L2_REG).to(device)
    aux_criterion = BCEWithLogitsLoss().to(device)
    
    optimizer = eval('optim.{}'.format(OPTIMIZER))(model.parameters(), lr=LEARNING_RATE)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_NUM, sampler=StratifiedSampler(torch.from_numpy(train_dataset.e.squeeze()), batch_size=BATCH_NUM), drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
    
    best_epoch_val_loss = 99999
    best_epoch_test_loss = 99999
    best_c_index = 0
    best_res = dict()
    
    flag = 0
    
    for epoch in range(1, EPOCHS + 1) :
        
        train_losses, train_neg_losses, train_aux_losses, train_reg_losses = [], [], [], []
        
        model.train()
        for X, y, e in tqdm(train_loader) :
            
            risk_pred = model(X.to(device), X.shape[0])
            
            train_neg_loss = criterion(risk_pred, y.to(device), e.to(device), device)
            train_reg_loss = reg_criterion(model)
            
            weighted_loss = train_neg_loss + train_reg_loss
            
            optimizer.zero_grad()
            weighted_loss.backward()
            optimizer.step()
            
            train_losses.append(weighted_loss.item())
            train_neg_losses.append(train_neg_loss.item())
            train_reg_losses.append(train_reg_loss.item())
            
        model.eval()
        with torch.no_grad() : 
            for X, y, e in val_loader :
                
                risk_pred = model(X.to(device), X.shape[0])
                
                val_neg_loss = criterion(risk_pred, y.to(device), e.to(device), device)
                val_reg_loss = reg_criterion(model)
                
                val_loss = val_neg_loss + val_reg_loss
                
                val_c = calculate_c_index(-risk_pred, y, e)
                                    
            for X, y, e in test_loader :
                
                risk_pred, lv_12_att, lv_22_att, lv_23_att = model(X.to(device), X.shape[0], return_attention_weights=True)
                                
                test_neg_loss = criterion(risk_pred, y.to(device), e.to(device), device)
                test_reg_loss = reg_criterion(model)

                test_loss = test_neg_loss + test_reg_loss
                
                test_c = calculate_c_index(-risk_pred, y, e)
                
        train_loss = np.average(train_losses)
        train_neg_loss = np.average(train_neg_losses)
        train_reg_loss = np.average(train_reg_losses)
        
        if best_epoch_val_loss > val_neg_loss :
            
            if not os.path.exists(os.path.join(os.getcwd(), "saved_models")) :
                os.makedirs(os.path.join(os.getcwd(), "saved_models"), exist_ok=True)
                
            torch.save(model.state_dict(), os.path.join(os.getcwd(), "saved_models", f"Assay={TARGET_GENE_SET}.LengthCutoff={LENGTH_RESTRICTION}.SEED={SEED}.SPLIT={SPLIT}.pt"))
            torch.save(lv_12_att, os.path.join(os.getcwd(), "saved_models", f"Assay={TARGET_GENE_SET}.LengthCutoff={LENGTH_RESTRICTION}.SEED={SEED}.SPLIT={SPLIT}.Attention.Lv12.pt"))
            torch.save(lv_22_att, os.path.join(os.getcwd(), "saved_models", f"Assay={TARGET_GENE_SET}.LengthCutoff={LENGTH_RESTRICTION}.SEED={SEED}.SPLIT={SPLIT}.Attention.Lv22.pt"))
            torch.save(lv_23_att, os.path.join(os.getcwd(), "saved_models", f"Assay={TARGET_GENE_SET}.LengthCutoff={LENGTH_RESTRICTION}.SEED={SEED}.SPLIT={SPLIT}.Attention.Lv23.pt"))
            torch.save(risk_pred, os.path.join(os.getcwd(), "saved_models", f"Assay={TARGET_GENE_SET}.LengthCutoff={LENGTH_RESTRICTION}.SEED={SEED}.SPLIT={SPLIT}.RiskPred.pt"))
            
            best_epoch_val_loss = val_neg_loss
            best_epoch_val_c = val_c
            best_epoch_test_loss = test_neg_loss
            best_epoch_test_c = test_c
            flag = 0        

        else :
            flag += 1
            if flag >= PATIENCE :
                return best_epoch_val_loss, best_epoch_val_c, best_epoch_test_loss, best_epoch_test_c
    
        print(f'\rEpoch: {epoch}' +
              f'\tTotalLoss (train-val-test) : {train_loss.item():.3f}, {val_loss.item():.3f}, {test_loss.item():.3f}' +
              f'\tTrainLoss (neg-reg) : {train_neg_loss.item():.3f}, {train_reg_loss.item():.3f}' +
              f'\tNegLoss (val-test) : {val_neg_loss.item():.3f}, {test_neg_loss.item():.3f}' +
              f'\tC-I : ({val_c:.3f}, {test_c:.3f})', end='', flush=True)
        
    return best_epoch_val_loss, best_epoch_val_c, best_epoch_test_loss, best_epoch_test_c


if __name__ == "__main__" : 
    
    args = argument_parsing()
    
    HYPERPARAMETERS = [f"SOURCE={args.SOURCE_GENE_SET}", 
                       f"TARGET={args.TARGET_GENE_SET}", 
                       f"BLOCK={args.BLOCK_CUTOFF}", 
                       f"BRIDGE={args.BRIDGE_CUTOFF}",
                       f"EVIDENCE={args.EVIDENCE_CUTOFF}"]

    print()
    print("\n".join(HYPERPARAMETERS))
    print()
    print()

    HYPERPARAMETERS = ".".join(HYPERPARAMETERS)
    
    SOURCE_GENE_SET = args.SOURCE_GENE_SET
    TARGET_GENE_SET = args.TARGET_GENE_SET
    BLOCK_CUTOFF = args.BLOCK_CUTOFF
    BRIDGE_CUTOFF = args.BRIDGE_CUTOFF
    EVIDENCE_CUTOFF = args.EVIDENCE_CUTOFF
    
    LENGTH_RESTRICTION = args.LENGTH_CUTOFF
    
    
    DATASET = "SCAN-B"
    FILTER_LOW_EXPRESSED_GENES = True
    
    SUBPATHWAY_FOLDER_PATH = os.path.join(os.getcwd(), f"SubpathwayCascade", HYPERPARAMETERS)
    MAPPED_PATH = os.path.join(SUBPATHWAY_FOLDER_PATH, f"MAPPING={DATASET}.LENGTH={LENGTH_RESTRICTION}")

    KG_dict_torch = torch.load(os.path.join(MAPPED_PATH, "HierarchicalKnowledgeGraph.Condensed.TorchObject.pt"))

    level_1_nodes = KG_dict_torch["level_1_nodes"]
    level_2_nodes = KG_dict_torch["level_2_nodes"]
    level_3_nodes = KG_dict_torch["level_3_nodes"]

    level_1_node_number = KG_dict_torch["level_1_node_number"]
    level_2_node_number = KG_dict_torch["level_2_node_number"]
    level_3_node_number = KG_dict_torch["level_3_node_number"]

    level_11_G_torch = KG_dict_torch["level_11_G_torch"]
    level_22_G_torch = KG_dict_torch["level_22_G_torch"]

    level_21_indicator = KG_dict_torch["level_21_indicator"]
    level_32_indicator = KG_dict_torch["level_32_indicator"]
    level_13_indicator = KG_dict_torch["level_13_indicator"]

    
    EPOCHS = args.EPOCHS
    DEVICE = args.DEVICE
    BATCH_NUM = args.BATCH_SIZE    
    HIDDEN_DIM = args.HIDDEN_DIM
    LEARNING_RATE = args.LEARNING_RATE
    PATIENCE = args.PATIENCE
    
    OPTIMIZER = "Adam"
    INPUT_DIM = 1
    GNN_DROPOUT = 0.2
    SURV_DROPOUT = 0.2
    LEVEL_3_AGG = "mean"
    AGG_STRATEGY = "concat"
    SURV_ACTIVATION = "ReLU"
    L2_REG = False
    
    msigdb_hallmarks = pd.read_csv("MSigDB_Hallmark_2020.txt", header=None, sep="\t")
    msigdb_hallmarks.set_index(0, inplace=True)
    msigdb_hallmarks.index.rename("Hallmark", inplace=True)
    msigdb_hallmark_genes = defaultdict(set)
    for idx, row in msigdb_hallmarks.iterrows() : msigdb_hallmark_genes[idx] = set(row.dropna())
    ATTRIBUTES = torch.from_numpy(gene_one_hot_encoding(msigdb_hallmark_genes.values(), msigdb_hallmark_genes.keys(), level_1_nodes).to_numpy().T).float()

    
    RANDOM_SEED = 0
    FOLD = 10

    cohort_surv = pd.read_csv(os.path.join(os.getcwd(), f"{DATASET}.EarlyStageBRCA.Survival.txt"), sep="\t")
    cohort_surv.set_index("GEX.assay", inplace=True)

    exp = pd.read_csv(os.path.join(os.getcwd(), f"{DATASET}.Length={LENGTH_RESTRICTION}.{HYPERPARAMETERS}.EarlyStageBRCA.Expression.txt"), sep="\t")
    exp.set_index("Unnamed: 0", inplace=True)
    exp = exp.groupby("Unnamed: 0").sum()

    with open(os.path.join(os.getcwd(), f'FOLD={FOLD}.SEED={RANDOM_SEED:04d}.pkl'), 'rb') as f : split_dict = pickle.load(f)

    print(f"Samples : {len(cohort_surv.index)}")
    
    
    result_log = defaultdict(list)
    for split_number, sample_split in split_dict.items() :

        X_train = exp[sample_split["train"]].T[level_1_nodes].to_numpy()
        X_val = exp[sample_split["val"]].T[level_1_nodes].to_numpy()
        X_test = exp[sample_split["test"]].T[level_1_nodes].to_numpy()

        X_train = np.log2(X_train+1)
        X_val = np.log2(X_val+1)
        X_test = np.log2(X_test+1)

        e_train = cohort_surv.loc[sample_split["train"]]["RFi_event"]
        e_val = cohort_surv.loc[sample_split["val"]]["RFi_event"]
        e_test = cohort_surv.loc[sample_split["test"]]["RFi_event"]

        y_train = cohort_surv.loc[sample_split["train"]]["RFi_days"]
        y_val = cohort_surv.loc[sample_split["val"]]["RFi_days"]
        y_test = cohort_surv.loc[sample_split["test"]]["RFi_days"]

        scaler = StandardScaler()
        scaler.fit(X_train)

        X_train = scaler.transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

        SPLIT = split_number
        
        train_dataset = ExpressionDataset(X_train, 
                                          np.expand_dims(e_train.to_numpy(), 1), 
                                          np.expand_dims(y_train.to_numpy(), 1))
        val_dataset = ExpressionDataset(X_val, 
                                        np.expand_dims(e_val.to_numpy(), 1), 
                                        np.expand_dims(y_val.to_numpy(), 1))
        test_dataset = ExpressionDataset(X_test, 
                                         np.expand_dims(e_test.to_numpy(), 1),
                                         np.expand_dims(y_test.to_numpy(), 1))

        best_epoch_val_loss, best_epoch_val_c, best_epoch_test_loss, best_epoch_test_c = train(train_dataset, val_dataset, test_dataset, DEVICE)

        print()
        print()
        
        print(f"Split : {split_number}")
        print(f"Best Epoch Val Loss : {best_epoch_val_loss:.3f}, Best Epoch Val C-Index : {best_epoch_val_c:.3f}, Best Epoch Test Loss : {best_epoch_test_loss:.3f}, Best Epoch Test C-Index : {best_epoch_test_c:.3f}")
        
        result_log["best_epoch_val_loss"].append(best_epoch_val_loss)
        result_log["best_epoch_val_c"].append(best_epoch_val_c)
        result_log["best_epoch_test_loss"].append(best_epoch_test_loss)
        result_log["best_epoch_test_c"].append(best_epoch_test_c)
