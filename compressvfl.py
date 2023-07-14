#!/usr/bin/env python
# coding: utf-8

# Afsana Khan, Marijn ten Thij, and Anna Wilbik. 2022. Communication-Efficient Vertical Federated Learning. Algorithms 15, 8 (2022)
# 
# Re-implemented by
# yoshitaka.inoue@docomoinnovations.com
# 2023.06.30
                                              

import os
import time
import math
import random
import pickle
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_auc_score, f1_score
from tqdm import tqdm
from sklearn.decomposition import PCA

import argparse



parser = argparse.ArgumentParser()
parser.add_argument('--data_dir',    type=str, default='data/adult3', help='')
parser.add_argument('--model_dir',   type=str, default='models', help='')
parser.add_argument('--log_dir',   type=str, default='runs', help='')
parser.add_argument('--log_comment', type=str, default='', help='')
parser.add_argument('--gpu_id',      type=int, default=0, help='')
parser.add_argument('--seed',        type=int, default=42, help='')

parser.add_argument('--model_header', type=str, default='compress_vfl', help='')
parser.add_argument('--clients',      type=str, default='0,1,2', help='')

parser.add_argument('--interface_dims',       type=str, default='8,8,8', help='')
parser.add_argument('--lr',         type=float, default=0.01, help='')
parser.add_argument('--batch_size', type=int, default=1024, help='')
parser.add_argument('--n_epochs',   type=int, default=200, help='')
parser.add_argument('--early_stopping', action='store_true')
parser.add_argument('--patience',   type=int, default=3, help='')
parser.add_argument('--tsv_name', type=str, default='result.tsv', help='')

# Statistics
parser.add_argument('--sparse_embed_epsilon', type=float, default=0, help='Threshold to consider sparse')
parser.add_argument('--sparse_grad_epsilon', type=float, default=0, help='Threshold to consider sparse')

# Model
parser.add_argument('--client_activation', type=str, default='relu', help='')
parser.add_argument('--client_optimizer', type=str, default='adam', help='')
parser.add_argument('--out_size', type=int, default=1, help='Binary: 1, Multi-class: N')
parser.add_argument('--loss', type=str, default='bcewll', help='bcewll, ce')

# Data reduction
parser.add_argument('--reduction', type=str, default='org', help='org, q8, q16')

# Preprocessing (compress by PCA or Auto-encoder)
parser.add_argument('--compress', type=str, default='pca', help='pca, ae')


args = parser.parse_args()
print(args)


data_dir = args.data_dir
model_dir = args.model_dir

#############################################
# Dataset: Feature, Label, indices, feature_column, label_column
#############################################
tr_feature = pickle.load(open(f'{data_dir}/tr_feature_v3.pkl', 'rb'))
va_feature = pickle.load(open(f'{data_dir}/va_feature_v3.pkl', 'rb'))

tr_label = pickle.load(open(f'{data_dir}/tr_label_v3.pkl', 'rb'))
va_label = pickle.load(open(f'{data_dir}/va_label_v3.pkl', 'rb'))

tr_indices = pickle.load(open(f'{data_dir}/tr_indices_v3.pkl', 'rb'))
va_indices = pickle.load(open(f'{data_dir}/va_indices_v3.pkl', 'rb'))

tr_f_col = pickle.load(open(f'{data_dir}/tr_f_col_v3.pkl', 'rb'))
va_f_col = pickle.load(open(f'{data_dir}/va_f_col_v3.pkl', 'rb'))

tr_l_col = pickle.load(open(f'{data_dir}/tr_l_col_v3.pkl', 'rb'))
va_l_col = pickle.load(open(f'{data_dir}/va_l_col_v3.pkl', 'rb'))


for tr_f in tr_feature:
    print(tr_f.shape)
for tr_l in tr_label:
    print(tr_l.shape)

class Dataset3(Dataset):
    def __init__(self, feature_list, label, indices, xcols_list, ycols):
        
        self.client_count = len(feature_list)
        
        for i in range(self.client_count):
            if i==0:
                f = feature_list[i][xcols_list[i]]
            else:
                f = pd.merge(
                    f,
                    feature_list[i][xcols_list[i]],
                    left_index=True, right_index=True,
                    how = 'left'
                ).fillna(0)
        
        self.x_list = []
        self.xcols_list = []
        for i in range(self.client_count):
            self.x_list.append(torch.Tensor(f[xcols_list[i]].values))
            self.xcols_list.append(xcols_list[i])
            
        self.y = torch.Tensor(label[ycols].values)
        self.indices = indices
        self.ycols = ycols
        
        # for binary-class BCEWithLogitLoss
        self.pos_weight = (self.y.shape[0] - self.y.sum()) / self.y.sum() # n0/n1 as scalar
        
        # for multi-class CrossEntropyLoss
        weight_table = label.value_counts().reset_index().sort_values(by=ycols[0])
        weight_table.columns = [ycols[0], 'weight']
        weight_table.weight = label.shape[0]/weight_table.weight
        self.weight = torch.Tensor(weight_table.weight.values)

    def __len__(self):
        return self.x1.shape[0]

    def __getitem__(self, idx):
        x = []
        for i in range(self.client_count):
            x.append(self.x_list[i][idx,:])
        return x, self.y[idx,:], self.indices[idx]


    def get_xcols(self, i):
        return self.xcols_list[i]
    def get_ycols(self):
        return self.ycols


client_id_list = [int(i) for i in args.clients.split(',')]


ds_tr = Dataset3(tr_feature, tr_label[client_id_list[0]], tr_indices[client_id_list[0]], tr_f_col, tr_l_col[client_id_list[0]])
ds_va = Dataset3(va_feature, va_label[client_id_list[0]], va_indices[client_id_list[0]], va_f_col, va_l_col[client_id_list[0]])

tr_uid = ds_tr.indices
tr_x1 = ds_tr.x_list[client_id_list[0]]
tr_x2 = ds_tr.x_list[client_id_list[1]]
tr_x3 = ds_tr.x_list[client_id_list[2]]
tr_xcols1 = ds_tr.xcols_list[client_id_list[0]]
tr_xcols2 = ds_tr.xcols_list[client_id_list[1]]
tr_xcols3 = ds_tr.xcols_list[client_id_list[2]]
tr_y = ds_tr.y

if args.loss=='bcewll':
    pos_weight = ds_tr.pos_weight
elif args.loss=='ce':
    weight=ds_tr.weight
else:
    print('error')


va_uid = ds_va.indices
va_x1 = ds_va.x_list[client_id_list[0]]
va_x2 = ds_va.x_list[client_id_list[1]]
va_x3 = ds_va.x_list[client_id_list[2]]
va_xcols1 = ds_va.xcols_list[client_id_list[0]]
va_xcols2 = ds_va.xcols_list[client_id_list[1]]
va_xcols3 = ds_va.xcols_list[client_id_list[2]]
va_y = ds_va.y







#############################################
# Data reduction
#############################################


def reduce_encode_emb(emb, cid, header='tr'):
    pos = None
    theoretical_size = 0
    if args.reduction=='org':
        ec = emb.detach().cpu()
        theoretical_size = (emb.shape[0] * emb.shape[1]) * 32
    elif args.reduction=='q4':
        scale = emb.max()
        ec = {'scale': scale.detach().cpu(), 'emb': (emb/scale*16).to(torch.quint4x2).detach().cpu()}
        theoretical_size = (emb.shape[0] * emb.shape[1]) * 4
    elif args.reduction=='q8':
        scale = emb.max()
        ec = {'scale': scale.detach().cpu(), 'emb': (emb/scale*255).byte().detach().cpu()}
        theoretical_size = (emb.shape[0] * emb.shape[1]) * 8
    elif args.reduction=='q16':
        ec = emb.to(torch.float16).detach().cpu()
        theoretical_size = (emb.shape[0] * emb.shape[1]) * 16
    else:
        print('error')
        
    torch.save(ec, f'{comm_dir}/{header}_emb_{cid}.pt')
    theoretical_size = math.ceil(theoretical_size/8.0) # bit to byte
    return pos, theoretical_size

def reduce_decode_emb(cid, header='tr'):
    e = torch.load(f'{comm_dir}/{header}_emb_{cid}.pt')
    nz_pos = None
    if args.reduction=='org':
        e = e.to(device)
    elif args.reduction=='q4':
        scale = e['scale'].to(device)
        e     = e['emb'].to(device)
        e = e.float() * scale / 16.0
    elif args.reduction=='q8':
        scale = e['scale'].to(device)
        e     = e['emb'].to(device)
        e = e.float() * scale / 255.0
    elif args.reduction=='q16':
        e = e.float().to(device) 
    else:
        print('error')
        e = None

    return e, nz_pos



#############################################
# GPU
#############################################

gpu_id = args.gpu_id
device = torch.device("cuda:" + str(gpu_id)) if torch.cuda.is_available() else torch.device("cpu")
print(device)


#############################################
# Setup VFL
#############################################
model_header = args.model_header
comm_dir = f'comm_{model_header}'
os.makedirs(comm_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

# Server data
print(tr_uid) # UserID, Common
print(tr_y.shape)
if args.loss=='bcewll':
    print(pos_weight)
elif args.loss=='ce':
    print(weight)
else:
    print('error')
print(va_uid) # UserID, Common
print(va_y.shape)


# Client 1 data
print(tr_uid) # UserID, Common
print(tr_x1.shape) # Data
print(tr_xcols1) # Columns
print(va_uid) # UserID, Common
print(va_x1.shape) # Data
print(va_xcols1) # Columns


# Client 2 data
print(tr_uid) #  UserID, Common
print(tr_x2.shape) # Data
print(tr_xcols2) # Columns
print(va_uid) #  UserID, Common
print(va_x2.shape) # Data
print(va_xcols2) # Columns


# Client 3 data
print(tr_uid) # UserID, Common
print(tr_x3.shape) # Data
print(tr_xcols3) # Columns
print(va_uid) # UserID, Common
print(va_x3.shape) # Data
print(va_xcols3) # Columns


# Server information
seed = args.seed # Random seed
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

early_stopping = args.early_stopping
patience = args.patience
patience_counter = 0
sparse_embed_epsilon = args.sparse_embed_epsilon
sparse_grad_epsilon  = args.sparse_grad_epsilon


# Common information
lr = args.lr
dim_1 = int(args.interface_dims.split(',')[client_id_list[0]])
dim_2 = int(args.interface_dims.split(',')[client_id_list[1]])
dim_3 = int(args.interface_dims.split(',')[client_id_list[2]])


# Information sent to clients from server
batch_size = args.batch_size
tr_shuffle = torch.randperm(len(tr_uid))
n_epochs = args.n_epochs


#############################################
# Models
#############################################
    
class ServerModel(torch.nn.Module):
    def __init__(self, hidden_size_list, out_size, hidden_layer_count=1):
        super(ServerModel, self).__init__()
        self.hidden_layer_count = hidden_layer_count
        self.client_count = len(hidden_size_list)
        hidden_size = sum(hidden_size_list)
        self.h2h, hidden_size = self._hidden_layers(hidden_size)
        self.h2o = torch.nn.Linear(hidden_size, out_size)   
        
        
        # Reproducability
        set_seed(seed)
        torch.nn.init.xavier_uniform_(self.h2o.weight.data)
        torch.nn.init.ones_(self.h2o.bias.data)
                
        
    def _hidden_layers(self, hidden_size):
        layers = []
        for i in range(self.hidden_layer_count):
            h2h = torch.nn.Linear(hidden_size, hidden_size//2)
            layers.append(h2h)
            # Reproducability
            set_seed(seed)
            torch.nn.init.xavier_uniform_(h2h.weight.data)
            torch.nn.init.ones_(h2h.bias.data)

            layers.append(torch.nn.ReLU())
            hidden_size = hidden_size//2
        
        return torch.nn.Sequential(*layers), hidden_size
        
    def forward(self, h):
        h = self.h2h(h)
        output = self.h2o(h)
        return output


#############################################
# Training
#############################################
###################
# Server
###################
server_model = ServerModel([dim_1, dim_2, dim_3], args.out_size, 1)
server_model = server_model.to(device)
optimizer = torch.optim.Adam(server_model.parameters(), lr=lr)
if args.loss=='bcewll':
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
elif args.loss=='ce':
    celoss = torch.nn.CrossEntropyLoss(weight=weight.to(device))
    def criterion(output, target):
        return celoss(output, target.flatten().long())
else:
    print('error')

best_va_score = -1


# Statistics

start_time = time.time()
writer = SummaryWriter(comment=args.log_comment)

total_count_send_1 = 0
total_count_send_2 = 0
total_count_send_3 = 0
total_count_receive_1 = 0
total_count_receive_2 = 0
total_count_receive_3 = 0

# Practical size
total_size_send_1 = 0
total_size_send_2 = 0
total_size_send_3 = 0
total_size_receive_1 = 0
total_size_receive_2 = 0
total_size_receive_3 = 0

# Theoretical size
total_size_send_1t = 0
total_size_send_2t = 0
total_size_send_3t = 0
total_size_receive_1t = 0
total_size_receive_2t = 0
total_size_receive_3t = 0

total_time_1 = 0
total_time_2 = 0
total_time_3 = 0
total_time_s = 0



# Send all the compressed data at once
###################
# Compress
###################
print(f'Compression method: {args.compress}')
# tr_x1, va_x1
# tr_x2, va_x2
# tr_x3, va_x3
# At the inference steps, te_x1, te_x2, te_x3

# Practical size
epoch_size_send_1 = 0
epoch_size_send_2 = 0
epoch_size_send_3 = 0
epoch_size_receive_1 = 0
epoch_size_receive_2 = 0
epoch_size_receive_3 = 0

# Theoretical size
epoch_size_send_1t = 0
epoch_size_send_2t = 0
epoch_size_send_3t = 0
epoch_size_receive_1t = 0
epoch_size_receive_2t = 0
epoch_size_receive_3t = 0

if args.compress == 'pca':
    ###################
    # Client
    ###################
    pca_1 = PCA(n_components=dim_1, svd_solver='full')
    pca_2 = PCA(n_components=dim_2, svd_solver='full')
    pca_3 = PCA(n_components=dim_3, svd_solver='full')
    
    tr_emb_1 = pca_1.fit_transform(tr_x1.numpy())
    tr_emb_2 = pca_2.fit_transform(tr_x2.numpy())
    tr_emb_3 = pca_3.fit_transform(tr_x3.numpy())
    
    va_emb_1 = pca_1.transform(va_x1.numpy())
    va_emb_2 = pca_2.transform(va_x2.numpy())
    va_emb_3 = pca_3.transform(va_x3.numpy())
    
    tr_emb_1 = torch.FloatTensor(tr_emb_1)
    tr_emb_2 = torch.FloatTensor(tr_emb_2)
    tr_emb_3 = torch.FloatTensor(tr_emb_3)
    
    va_emb_1 = torch.FloatTensor(va_emb_1)
    va_emb_2 = torch.FloatTensor(va_emb_2)
    va_emb_3 = torch.FloatTensor(va_emb_3)

    _, size_e1t = reduce_encode_emb(tr_emb_1, 1, 'tr')
    _, size_e2t = reduce_encode_emb(tr_emb_2, 2, 'tr')
    _, size_e3t = reduce_encode_emb(tr_emb_3, 3, 'tr')
    epoch_size_send_1t += size_e1t
    epoch_size_send_2t += size_e2t
    epoch_size_send_3t += size_e3t
    epoch_size_send_1 += os.path.getsize(f'{comm_dir}/tr_emb_1.pt')
    epoch_size_send_2 += os.path.getsize(f'{comm_dir}/tr_emb_2.pt')
    epoch_size_send_3 += os.path.getsize(f'{comm_dir}/tr_emb_3.pt')
    
    _, size_e1t = reduce_encode_emb(va_emb_1, 1, 'va')
    _, size_e2t = reduce_encode_emb(va_emb_2, 2, 'va')
    _, size_e3t = reduce_encode_emb(va_emb_3, 3, 'va')
    epoch_size_send_1t += size_e1t
    epoch_size_send_2t += size_e2t
    epoch_size_send_3t += size_e3t
    epoch_size_send_1 += os.path.getsize(f'{comm_dir}/va_emb_1.pt')
    epoch_size_send_2 += os.path.getsize(f'{comm_dir}/va_emb_2.pt')
    epoch_size_send_3 += os.path.getsize(f'{comm_dir}/va_emb_3.pt')

    ###################
    # Server
    ###################
    tr_x1, _ = reduce_decode_emb(1, header='tr')
    tr_x2, _ = reduce_decode_emb(2, header='tr')
    tr_x3, _ = reduce_decode_emb(3, header='tr')
    va_x1, _ = reduce_decode_emb(1, header='va')
    va_x2, _ = reduce_decode_emb(2, header='va')
    va_x3, _ = reduce_decode_emb(3, header='va')

elif args.compress == 'ae':
    class AutoEncoder(torch.nn.Module):
        def __init__(self, in_size, hidden_size):
            super().__init__()
            
            # Reproducability

            self.encoder = torch.nn.Sequential(
                torch.nn.Linear(in_size, in_size//2),
                torch.nn.ReLU(),
                torch.nn.Linear(in_size//2, hidden_size),
                torch.nn.ReLU(),
            )

            self.decoder = torch.nn.Sequential(
                torch.nn.Linear(hidden_size, in_size//2),
                torch.nn.ReLU(),
                torch.nn.Linear(in_size//2, in_size),
            )

            # Reproducability
            set_seed(seed)
            torch.nn.init.xavier_uniform_(self.encoder[0].weight.data)
            torch.nn.init.ones_(self.encoder[0].bias.data)
            set_seed(seed)
            torch.nn.init.xavier_uniform_(self.encoder[2].weight.data)
            torch.nn.init.ones_(self.encoder[2].bias.data)
            set_seed(seed)
            torch.nn.init.xavier_uniform_(self.decoder[0].weight.data)
            torch.nn.init.ones_(self.decoder[0].bias.data)
            set_seed(seed)
            torch.nn.init.xavier_uniform_(self.decoder[2].weight.data)
            torch.nn.init.ones_(self.decoder[2].bias.data)

            
        def forward(self, x):
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
            return decoded


    def train_ae(tr_x, va_x, cid, in_size, dim):
        ae = AutoEncoder(in_size, dim)
        ae = ae.to(device)
        loss_function = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(ae.parameters(), lr=lr)
        
        best_va_loss = 99999999
        for epoch in range(n_epochs):
            
            ####################
            # Traininig
            ####################
            tr_sample_count = len(tr_uid)
            tr_batch_count = math.ceil(tr_sample_count / batch_size)
            ae.train()
            tr_loss = 0
            
            for batch_index in tqdm(range(tr_batch_count)):
                head = batch_index * batch_size
                tail = min(head + batch_size, tr_sample_count)
                si = tr_shuffle[head:tail]
                batch_uid = tr_uid[si]
                batch_x = tr_x[si,:].to(device)

                decoded = ae(batch_x)
                loss = loss_function(decoded, batch_x)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                tr_loss += loss.item()
                
            ####################
            # Validation
            ####################
            va_sample_count = len(va_uid)
            va_batch_count = math.ceil(va_sample_count / batch_size)
            ae.eval()
            va_loss = 0
            for batch_index in tqdm(range(va_batch_count)):
                head = batch_index * batch_size
                tail = min(head + batch_size, va_sample_count)
                batch_uid = va_uid[head:tail]
                batch_x = va_x[head:tail,:].to(device)

                decoded = ae(batch_x)
                loss = loss_function(decoded, batch_x)
                va_loss += loss.item()
            
            ##########
            tr_loss /= tr_batch_count
            va_loss /= va_batch_count
                
            if best_va_loss > va_loss:
                best_va_loss = va_loss
                model_name = f'{model_dir}/{model_header}_ae_{cid}_model.pt'
                torch.save(ae.state_dict(), model_name)
                patience_counter = 0
            else:
                if True: # Force early stopping for AutoEncoder
                    patience_counter += 1
                    print(f'Waiting early stop {patience_counter}/{patience}')
                    if patience_counter > patience:
                        print('stopped')
                        break
            print(f'epoch: {epoch}/{n_epochs}, tr_loss: {tr_loss}, va_loss: {va_loss}')
        
        print('Load best autoencoder')
        ae.load_state_dict(torch.load(f'{model_dir}/{model_header}_ae_{cid}_model.pt'))
        return ae
        
                        
    def infer_ae(x, ae, sample_count, batch_count, dim):
        emb = torch.zeros(sample_count, dim)
        ae = ae.to(device)
        ae.eval()
        for batch_index in tqdm(range(batch_count)):
            head = batch_index * batch_size
            tail = min(head + batch_size, sample_count)
            batch_x = x[head:tail,:].to(device)
            hidden = ae.encoder(batch_x)
            emb[head:tail,:] = hidden.detach().cpu()
            
        return emb
    
    print('Training AutoEncoder 1')
    ae_1 = train_ae(tr_x1, va_x1, 1, len(tr_xcols1), dim_1)
    print('Training AutoEncoder 2')
    ae_2 = train_ae(tr_x2, va_x2, 2, len(tr_xcols2), dim_2)
    print('Training AutoEncoder 3')
    ae_3 = train_ae(tr_x3, va_x3, 3, len(tr_xcols3), dim_3)

    tr_sample_count = len(tr_uid)
    tr_batch_count = math.ceil(tr_sample_count / batch_size)
    tr_emb_1 = infer_ae(tr_x1, ae_1, tr_sample_count, tr_batch_count, dim_1)
    tr_emb_2 = infer_ae(tr_x2, ae_2, tr_sample_count, tr_batch_count, dim_2)
    tr_emb_3 = infer_ae(tr_x3, ae_3, tr_sample_count, tr_batch_count, dim_3)
    
    va_sample_count = len(va_uid)
    va_batch_count = math.ceil(va_sample_count / batch_size)
    va_emb_1 = infer_ae(va_x1, ae_1, va_sample_count, va_batch_count, dim_1)
    va_emb_2 = infer_ae(va_x2, ae_2, va_sample_count, va_batch_count, dim_2)
    va_emb_3 = infer_ae(va_x3, ae_3, va_sample_count, va_batch_count, dim_3)

    _, size_e1t = reduce_encode_emb(tr_emb_1, 1, 'tr')
    _, size_e2t = reduce_encode_emb(tr_emb_2, 2, 'tr')
    _, size_e3t = reduce_encode_emb(tr_emb_3, 3, 'tr')
    epoch_size_send_1t += size_e1t
    epoch_size_send_2t += size_e2t
    epoch_size_send_3t += size_e3t
    epoch_size_send_1 += os.path.getsize(f'{comm_dir}/tr_emb_1.pt')
    epoch_size_send_2 += os.path.getsize(f'{comm_dir}/tr_emb_2.pt')
    epoch_size_send_3 += os.path.getsize(f'{comm_dir}/tr_emb_3.pt')
    
    _, size_e1t = reduce_encode_emb(va_emb_1, 1, 'va')
    _, size_e2t = reduce_encode_emb(va_emb_2, 2, 'va')
    _, size_e3t = reduce_encode_emb(va_emb_3, 3, 'va')
    epoch_size_send_1t += size_e1t
    epoch_size_send_2t += size_e2t
    epoch_size_send_3t += size_e3t
    epoch_size_send_1 += os.path.getsize(f'{comm_dir}/va_emb_1.pt')
    epoch_size_send_2 += os.path.getsize(f'{comm_dir}/va_emb_2.pt')
    epoch_size_send_3 += os.path.getsize(f'{comm_dir}/va_emb_3.pt')

    ###################
    # Server
    ###################
    tr_x1, _ = reduce_decode_emb(1, header='tr')
    tr_x2, _ = reduce_decode_emb(2, header='tr')
    tr_x3, _ = reduce_decode_emb(3, header='tr')
    va_x1, _ = reduce_decode_emb(1, header='va')
    va_x2, _ = reduce_decode_emb(2, header='va')
    va_x3, _ = reduce_decode_emb(3, header='va')
    
total_size_send_1    += epoch_size_send_1
total_size_send_2    += epoch_size_send_2
total_size_send_3    += epoch_size_send_3
total_size_send_1t    += epoch_size_send_1t
total_size_send_2t    += epoch_size_send_2t
total_size_send_3t    += epoch_size_send_3t
total_size_all   = total_size_send_1  + total_size_send_2  + total_size_send_3  + total_size_receive_1  + total_size_receive_2  + total_size_receive_3
total_size_allt  = total_size_send_1t + total_size_send_2t + total_size_send_3t + total_size_receive_1t + total_size_receive_2t + total_size_receive_3t

###################
# Statistics
###################
tr_sparse_embed_loss = 0
tr_embed_sparsity = 0
tr_grad_sparsity = 0
    
    
for epoch in range(n_epochs):
    print(f'epoch: {epoch}/{n_epochs}')
    ##########################################################################
    # Traininig
    ##########################################################################
    print('Traininig')

    ###################
    # Common
    ###################
    tr_sample_count = len(tr_uid)
    tr_batch_count = math.ceil(tr_sample_count / batch_size)
    
    
    ###################
    # Server
    ###################
    server_model.train()
    
    tr_loss = 0
    tr_pred = np.zeros(tr_y.shape)
    tr_true = np.zeros(tr_y.shape)

    for batch_index in tqdm(range(tr_batch_count)):
        ###################
        # Server
        ###################
        head = batch_index * batch_size
        tail = min(head + batch_size, tr_sample_count)
        si = tr_shuffle[head:tail]
        batch_uid = tr_uid[si]

        st = time.time()
        batch_x_1 = tr_x1[si,:].to(device)
        batch_x_2 = tr_x2[si,:].to(device)
        batch_x_3 = tr_x3[si,:].to(device)
        batch_y   = tr_y[si,:].to(device)
        optimizer.zero_grad()
        emb = torch.cat((batch_x_1, batch_x_2, batch_x_3), 1)
        
        pred_y = server_model(emb)
        loss = criterion(pred_y, batch_y)
            
        loss.backward()
        optimizer.step()

        ## Training results
        tr_loss += loss.item()
            
        if args.loss=='bcewll':
            tr_pred[head:tail,:] = torch.sigmoid(pred_y).detach().cpu().numpy()
            tr_true[head:tail,:] = batch_y.detach().cpu().numpy()
        elif args.loss=='ce':
            tr_pred[head:tail,:] = torch.argmax(pred_y, dim=1, keepdim=True).detach().cpu().numpy()
            tr_true[head:tail,:] = batch_y.detach().cpu().numpy()
        else:
            print('error')

        total_time_s += (time.time() - st)
        
    
    
    ##########################################################################
    # Validation
    ##########################################################################
    print('Validation')
    
    ###################
    # Server
    ###################
    va_sample_count = len(va_uid)
    va_batch_count = math.ceil(va_sample_count / batch_size)

    server_model.eval()
    va_loss = 0
    va_pred = np.zeros(va_y.shape)
    va_true = np.zeros(va_y.shape)    
    
    for batch_index in tqdm(range(va_batch_count)):
        ###################
        # Server
        ###################
        head = batch_index * batch_size
        tail = min(head + batch_size, va_sample_count)
        batch_uid = va_uid[head:tail]

        st = time.time()
        batch_x_1 = va_x1[head:tail,:].to(device)
        batch_x_2 = va_x2[head:tail,:].to(device)
        batch_x_3 = va_x3[head:tail,:].to(device)
        batch_y   = va_y[head:tail,:].to(device)
        
        emb = torch.cat((batch_x_1, batch_x_2, batch_x_3), 1).to(device)
        pred_y = server_model(emb)
        loss = criterion(pred_y, batch_y)
        va_loss += loss.item()
        
        ## Validation results
        if args.loss=='bcewll':
            va_pred[head:tail,:] = torch.sigmoid(pred_y).detach().cpu().numpy()
            va_true[head:tail,:] = batch_y.detach().cpu().numpy()
        elif args.loss=='ce':
            va_pred[head:tail,:] = torch.argmax(pred_y, dim=1, keepdim=True).detach().cpu().numpy()
            va_true[head:tail,:] = batch_y.detach().cpu().numpy()
        else:
            print('error')
            
        total_time_s += (time.time() - st)


    
    # total Loss
    tr_loss /= tr_batch_count
    va_loss /= va_batch_count
    
    # auc
    if args.loss=='bcewll':
        tr_score = roc_auc_score(tr_true, tr_pred)
        va_score = roc_auc_score(va_true, va_pred)
        print(f'epoch: {epoch}/{n_epochs}, tr_loss: {tr_loss}, va_loss: {va_loss}, tr_auc: {tr_score}, va_auc: {va_score}')
        writer.add_scalar('tr/auc', tr_score, epoch)
        writer.add_scalar('va/auc', va_score, epoch)
    elif args.loss=='ce':
        tr_score = f1_score(tr_true, tr_pred, average='macro')
        va_score = f1_score(va_true, va_pred, average='macro')
        print(f'epoch: {epoch}/{n_epochs}, tr_loss: {tr_loss}, va_loss: {va_loss}, tr_macrof1: {tr_score}, va_macrof1: {va_score}')
        writer.add_scalar('tr/macrof1', tr_score, epoch)
        writer.add_scalar('va/macrof1', va_score, epoch)
    
    writer.add_scalar('tr/loss', tr_loss, epoch)
    writer.add_scalar('va/loss', va_loss, epoch)
    writer.flush()
    
    if va_score > best_va_score:
        best_va_score = va_score
        
        model_name = f'{model_dir}/{model_header}_server_model.pt'
        torch.save(server_model.state_dict(), model_name)
        patience_counter = 0        

    else:
        if early_stopping:
            patience_counter += 1
            print(f'Waiting early stop {patience_counter}/{patience}')
            if patience_counter > patience:
                print('stopped')
                break


end_time = time.time()
print(f'{end_time-start_time} [sec] for training')
    
writer.add_scalar('final/time', end_time-start_time)
writer.flush()



###################
# Best model
###################
print('Best model')
print(f'best_va_score: {best_va_score}')
print(f'{model_dir}/{model_header}_server_model.pt')

# Test data
te_feature = pickle.load(open(f'{data_dir}/te_feature_v3.pkl', 'rb'))
te_label   = pickle.load(open(f'{data_dir}/te_label_v3.pkl', 'rb'))
te_indices = pickle.load(open(f'{data_dir}/te_indices_v3.pkl', 'rb'))
te_f_col   = pickle.load(open(f'{data_dir}/te_f_col_v3.pkl', 'rb'))
te_l_col   = pickle.load(open(f'{data_dir}/te_l_col_v3.pkl', 'rb'))
ds_te = Dataset3(te_feature, te_label[0], te_indices[0], te_f_col, te_l_col[0])

te_uid = ds_te.indices
te_x1 = ds_te.x_list[0]
te_x2 = ds_te.x_list[1]
te_x3 = ds_te.x_list[2]
te_xcols1 = ds_te.xcols_list[0]
te_xcols2 = ds_te.xcols_list[1]
te_xcols3 = ds_te.xcols_list[2]
te_y = ds_te.y


###################
# Compress
###################
if args.compress == 'pca':
    ###################
    # Client
    ###################
    te_emb_1 = pca_1.transform(te_x1.numpy())
    te_emb_2 = pca_2.transform(te_x2.numpy())
    te_emb_3 = pca_3.transform(te_x3.numpy())
    
    te_emb_1 = torch.FloatTensor(te_emb_1)
    te_emb_2 = torch.FloatTensor(te_emb_2)
    te_emb_3 = torch.FloatTensor(te_emb_3)
    
    reduce_encode_emb(te_emb_1, 1, 'te')
    reduce_encode_emb(te_emb_2, 2, 'te')
    reduce_encode_emb(te_emb_3, 3, 'te')
    

    ###################
    # Server
    ###################
    te_x1, _ = reduce_decode_emb(1, header='te')
    te_x2, _ = reduce_decode_emb(2, header='te')
    te_x3, _ = reduce_decode_emb(3, header='te')

elif args.compress == 'ae':
    ###################
    # Client
    ###################
    
    print('Load best AutoEncoder model')
    ae_1.load_state_dict(torch.load(f'{model_dir}/{model_header}_ae_1_model.pt'))
    ae_1 = ae_1.to(device)
    ae_2.load_state_dict(torch.load(f'{model_dir}/{model_header}_ae_2_model.pt'))
    ae_2 = ae_2.to(device)
    ae_3.load_state_dict(torch.load(f'{model_dir}/{model_header}_ae_3_model.pt'))
    ae_3 = ae_3.to(device)


    te_sample_count = len(te_uid)
    te_batch_count = math.ceil(te_sample_count / batch_size)
    te_emb_1 = infer_ae(te_x1, ae_1, te_sample_count, te_batch_count, dim_1)
    te_emb_2 = infer_ae(te_x2, ae_2, te_sample_count, te_batch_count, dim_2)
    te_emb_3 = infer_ae(te_x3, ae_3, te_sample_count, te_batch_count, dim_3)
    
    reduce_encode_emb(te_emb_1, 1, 'te')
    reduce_encode_emb(te_emb_2, 2, 'te')
    reduce_encode_emb(te_emb_3, 3, 'te')
    

    ###################
    # Server
    ###################
    te_x1, _ = reduce_decode_emb(1, header='te')
    te_x2, _ = reduce_decode_emb(2, header='te')
    te_x3, _ = reduce_decode_emb(3, header='te')
    

###################
# Statistics
###################
te_embed_sparsity = 0


###################
# Server
###################
te_sample_count = len(te_uid)
te_batch_count = math.ceil(te_sample_count / batch_size)

server_model.load_state_dict(torch.load(f'{model_dir}/{model_header}_server_model.pt'))
server_model = server_model.to(device)
server_model.eval()
te_pred = np.zeros(te_y.shape)
te_true = np.zeros(te_y.shape)


for batch_index in tqdm(range(te_batch_count)):
    ###################
    # Server: Receive
    ###################
    head = batch_index * batch_size
    tail = min(head + batch_size, te_sample_count)        
    batch_uid = te_uid[head:tail]

    batch_x_1 = te_x1[head:tail,:].to(device)
    batch_x_2 = te_x2[head:tail,:].to(device)
    batch_x_3 = te_x3[head:tail,:].to(device)
    batch_y   = te_y[head:tail,:].to(device) # Label
    emb = torch.cat((batch_x_1, batch_x_2, batch_x_3), 1).to(device) # Concat horizontally
    pred_y = server_model(emb)
    
    if args.loss=='bcewll':
        te_pred[head:tail,:] = torch.sigmoid(pred_y).detach().cpu().numpy()
        te_true[head:tail,:] = batch_y.detach().cpu().numpy()
    elif args.loss=='ce':
        te_pred[head:tail,:] = torch.argmax(pred_y, dim=1, keepdim=True).detach().cpu().numpy()
        te_true[head:tail,:] = batch_y.detach().cpu().numpy()
    else:
        print('error')
    

###################
# Statistics
###################
if args.loss=='bcewll':
    te_score = roc_auc_score(te_true, te_pred)
    print(f'te_auc: {te_score:.4f}')
    writer.add_scalar('final/te_auc',  te_score)
elif args.loss=='ce':
    te_score = f1_score(te_true, te_pred, average='macro')
    print(f'te_macrof1: {te_score:.4f}')
    writer.add_scalar('final/te_macrof1',  te_score)

writer.flush()
writer.close()

result = pd.DataFrame({
    'log': [writer.log_dir.split('/')[1]],
    'batch_size': [args.batch_size],
    'client_activation': [args.client_activation],
    'client_optimizer': [args.client_optimizer],
    'clients': [args.clients],
    'data_dir': [args.data_dir],
    'early_stopping': [args.early_stopping],
    'gpu_id': [args.gpu_id],
    'interface_dims': [args.interface_dims],
    'layer_no_bias': [False],
    'log_comment': [args.log_comment],
    'loss': [args.loss],
    'lr': [args.lr],
    'model_dir': [args.model_dir],
    'model_header': [args.model_header],
    'n_epochs': [args.n_epochs],
    'out_size': [args.out_size],
    'patience': [args.patience],
    'reduction': [f'{args.compress}_{args.reduction}'],
    'seed': [args.seed],
    'sparse_embed_epsilon': [-1],
    'sparse_embed_lambda': [-1],
    'sparse_grad_epsilon': [-1],
    'top_k': [-1],
    
    'total_size_send_1': [total_size_send_1],
    'total_size_send_2': [total_size_send_2],
    'total_size_send_3': [total_size_send_3],
    'total_size_receive_1': [total_size_receive_1],
    'total_size_receive_2': [total_size_receive_2],
    'total_size_receive_3': [total_size_receive_3],
    'total_size_all': [total_size_all],

    'total_size_send_1t': [total_size_send_1t],
    'total_size_send_2t': [total_size_send_2t],
    'total_size_send_3t': [total_size_send_3t],
    'total_size_receive_1t': [total_size_receive_1t],
    'total_size_receive_2t': [total_size_receive_2t],
    'total_size_receive_3t': [total_size_receive_3t],
    'total_size_allt': [total_size_allt],

    
    'total_time_1': [-1],
    'total_time_2': [-1],
    'total_time_3': [-1],
    'total_time_s': [total_time_s],
    'te_score': [te_score],
    'te_embed_sparsity': [-1],
    'time': [end_time-start_time],
})

if os.path.exists(args.tsv_name):
    result_all = pd.read_csv(args.tsv_name, delimiter='\t')
    result_all = result_all.append(result)
    result_all.to_csv(args.tsv_name, sep='\t', index=False)
else:
    result.to_csv(args.tsv_name, sep='\t', index=False)


