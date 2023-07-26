#!/usr/bin/env python
# coding: utf-8


#   ____                          __     _______ _     
#  / ___| _ __   __ _ _ __ ___  __\ \   / /  ___| |    
#  \___ \| '_ \ / _` | '__/ __|/ _ \ \ / /| |_  | |    
#   ___) | |_) | (_| | |  \__ \  __/\ V / |  _| | |___ 
#  |____/| .__/ \__,_|_|  |___/\___| \_/  |_|   |_____|
#        |_|                                           
# Developed by
#  __  .__   __.   ______    __    __   _______ 
# |  | |  \ |  |  /  __  \  |  |  |  | |   ____|
# |  | |   \|  | |  |  |  | |  |  |  | |  |__   
# |  | |  . `  | |  |  |  | |  |  |  | |   __|  
# |  | |  |\   | |  `--'  | |  `--'  | |  |____ 
# |__| |__| \__|  \______/   \______/  |_______|
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

import argparse



parser = argparse.ArgumentParser()
parser.add_argument('--data_dir',    type=str, default='data/adult3', help='')
parser.add_argument('--data_format',   type=str, default='paper', help='paper, aws')
parser.add_argument('--model_dir',   type=str, default='models', help='')
parser.add_argument('--log_dir',   type=str, default='runs', help='')
parser.add_argument('--log_comment', type=str, default='', help='')
parser.add_argument('--gpu_id',      type=int, default=0, help='')
parser.add_argument('--seed',        type=int, default=42, help='')

parser.add_argument('--model_header', type=str, default='vfl_v7', help='')
parser.add_argument('--clients',      type=str, default='0,1,2', help='')

parser.add_argument('--interface_dims',       type=str, default='8,8,8', help='')
parser.add_argument('--lr',         type=float, default=0.01, help='')
parser.add_argument('--batch_size', type=int, default=1024, help='')
parser.add_argument('--n_epochs',   type=int, default=200, help='')
parser.add_argument('--early_stopping', action='store_true')
parser.add_argument('--patience',   type=int, default=3, help='')
parser.add_argument('--tsv_name', type=str, default='result.tsv', help='')

# L1-norm
parser.add_argument('--layer_no_bias', action='store_true')
parser.add_argument('--sparse_embed_lambda', type=float, default=0, help='Recommend: 0.1')

# Statistics
parser.add_argument('--sparse_embed_epsilon', type=float, default=0, help='Threshold to consider sparse')
parser.add_argument('--sparse_grad_epsilon', type=float, default=0, help='Threshold to consider sparse')

# Model
parser.add_argument('--client_activation', type=str, default='relu', help='')
parser.add_argument('--client_optimizer', type=str, default='adam', help='')
parser.add_argument('--out_size', type=int, default=1, help='Binary: 1, Multi-class: N')
parser.add_argument('--loss', type=str, default='bcewll', help='bcewll, ce')
parser.add_argument('--run_axis', type=int, default=1, help='Run-direction for run-length coding. Default: 1 (vertical)')
parser.add_argument('--norm', type=int, default=1, help='1 or 2')

# Data reduction (Our proposed algorithm is svfl16)
parser.add_argument('--reduction', type=str, default='org', help='org, int, q8, q16, topk, topk16, svfl, svfl16')
# Used for topk and topk16
parser.add_argument('--top_k', type=int, default=1024*4, help='')



args = parser.parse_args()
print(args)




data_dir = args.data_dir
model_dir = args.model_dir

#############################################
# Dataset: Feature, Label, indices, feature_column, label_column
#############################################
client_id_list = [int(i) for i in args.clients.split(',')]

if args.data_format == 'paper':
    '''
    Load dataset
    '''
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
    
    
elif args.data_format == 'aws':
    '''
    Load the same dataset as https://github.com/docomoinnovations/AWS-Serverless-Vertical-Federated-Learning/blob/main/init_data.py
    This code accepts only three clients and commented out client-4.
    '''
    
    print('Load the same dataset as AWS')

    # Common
    tr_uid = torch.LongTensor(np.load(f"{data_dir}/server/functions/init_server/tr_uid.npy", allow_pickle=False))
    va_uid = torch.LongTensor(np.load(f"{data_dir}/server/functions/init_server/va_uid.npy", allow_pickle=False))

    # Client
    tr_x1 = torch.FloatTensor(np.load(f"{data_dir}/client/dataset/client1/tr_x.npy", allow_pickle=False))
    tr_x2 = torch.FloatTensor(np.load(f"{data_dir}/client/dataset/client2/tr_x.npy", allow_pickle=False))
    tr_x3 = torch.FloatTensor(np.load(f"{data_dir}/client/dataset/client3/tr_x.npy", allow_pickle=False))
#     tr_x4 = torch.FloatTensor(np.load(f"{data_dir}/client/dataset/client4/tr_x.npy", allow_pickle=False))
    tr_xcols1 = np.load(f"{data_dir}/client/dataset/client1/cols.npy", allow_pickle=False)
    tr_xcols2 = np.load(f"{data_dir}/client/dataset/client2/cols.npy", allow_pickle=False)
    tr_xcols3 = np.load(f"{data_dir}/client/dataset/client3/cols.npy", allow_pickle=False)
#     tr_xcols4 = np.load(f"{data_dir}/client/dataset/client4/cols.npy", allow_pickle=False)
    va_x1 = torch.FloatTensor(np.load(f"{data_dir}/client/dataset/client1/va_x.npy", allow_pickle=False))
    va_x2 = torch.FloatTensor(np.load(f"{data_dir}/client/dataset/client2/va_x.npy", allow_pickle=False))
    va_x3 = torch.FloatTensor(np.load(f"{data_dir}/client/dataset/client3/va_x.npy", allow_pickle=False))
#     va_x4 = torch.FloatTensor(np.load(f"{data_dir}/client/dataset/client4/va_x.npy", allow_pickle=False))
    va_xcols1 = np.load(f"{data_dir}/client/dataset/client1/cols.npy", allow_pickle=False)
    va_xcols2 = np.load(f"{data_dir}/client/dataset/client2/cols.npy", allow_pickle=False)
    va_xcols3 = np.load(f"{data_dir}/client/dataset/client3/cols.npy", allow_pickle=False)
#     va_xcols4 = np.load(f"{data_dir}/client/dataset/client4/cols.npy", allow_pickle=False)


    # Server
    tr_y = torch.Tensor(np.load(f"{data_dir}/server/functions/server_training/tr_y.npy", allow_pickle=False))
    va_y = torch.Tensor(np.load(f"{data_dir}/server/functions/server_training/va_y.npy", allow_pickle=False))
    
    if args.loss=='bcewll':
        # for binary-class BCEWithLogitLoss
        pos_weight = (tr_y.shape[0] - tr_y.sum()) / tr_y.sum() # n0/n1 as scalar
    elif args.loss=='ce':
        # for multi-class CrossEntropyLoss
        label, label_count = np.unique(tr_y.flatten().numpy(), return_counts=True)
        weight = torch.Tensor(label_count.sum()/label_count)
    else:
        print('error')




#############################################
# Data reduction
#############################################
def sparse_encode(src, device, axis=1, target=0, get_position=0, nz_pos=None):
    samples = src.shape[0]
    dims = src.shape[1]
    
    # run-length compress by axis=1
    if axis==0:
        dst = src.reshape(-1)
    elif axis==1:
        dst = src.t().reshape(-1)
        
            
    if nz_pos is not None:
        zero_loc = ~nz_pos.to(device)
    else:
        zero_loc = (dst==0).to(device)
    non_zero_values = dst[~zero_loc].to(device) # to be sent
    
    #######################
    # Extract change points
    #######################
    zero_loc = zero_loc.char()
    length = len(zero_loc)
    ind = torch.arange(0, length).to(device)
        
    # none zero position
    nz_cp = torch.cat((torch.CharTensor([1]).to(device), zero_loc[0:-1]), 0).to(device) - zero_loc
    nz_head = ind[nz_cp==1]
    nz_tail = ind[nz_cp==-1]
    
    
    if get_position==0:
        # Fully encode
        info = {
            'samples': int(samples),
            'dims': int(dims),
            'length_non_zero_values': int(non_zero_values.shape[0]),
            'length_nz_head': int(nz_head.shape[0]),
            'non_zero_values': non_zero_values.float().detach().cpu(), # float32, tensor
            'nz_head': nz_head.int().detach().cpu(), # int32, tensor
            'nz_tail': nz_tail.int().detach().cpu(), # int32, tensor
        }
        return info
    elif get_position==1:
        # Without position
        info = {
            'samples': int(samples),
            'dims': int(dims),
            'length_non_zero_values': int(non_zero_values.shape[0]),
            'length_nz_head': int(nz_head.shape[0]),
            'non_zero_values': non_zero_values.float().detach().cpu(), # float32, tensor
        }
        return info
    elif get_position==2:
        # Duplicate position
        info = {
            'samples': int(samples),
            'dims': int(dims),
            'length_non_zero_values': int(non_zero_values.shape[0]),
            'length_nz_head': int(nz_head.shape[0]),
            'non_zero_values': non_zero_values.float().detach().cpu(), # float32, tensor
            'nz_head': nz_head.int().detach().cpu(), # int32, tensor
            'nz_tail': nz_tail.int().detach().cpu(), # int32, tensor
        }
        position = {
            'nz_head': nz_head, # long, tensor, gpu
            'nz_tail': nz_tail, # long, tensor, gpu
        }
        return info, position
        


def sparse_decode(info, device, axis=1, target=0, set_position=None):
    # Extract data
    samples = info['samples']
    dims    = info['dims']
    non_zero_values = info['non_zero_values'].to(device)
    if set_position is None:
        nz_head = info['nz_head'].long().to(device)
        nz_tail = info['nz_tail'].long().to(device)
    else:
        nz_head = set_position['nz_head'].long().to(device)
        nz_tail = set_position['nz_tail'].long().to(device)
    
    
    length = samples * dims
    nz_cp = torch.zeros(length).int().to(device)
    
    nz_cp[nz_head] = 1
    nz_cp[nz_tail] = -1
    
    nz_pos = torch.cumsum(nz_cp, dim=0).bool().to(device)
    
    dst = torch.zeros(length).to(device)
    dst[nz_pos] = non_zero_values
    
    if axis==0:
        dst = dst.reshape((samples, dims))
    elif axis==1:
        dst = dst.reshape((dims, samples)).t()
    
    return dst, nz_pos
    


def sparse_encode16(src, device, get_position=0, nz_pos=None, axis=1):
    # src (gpu, float32), nz_pos (cpu, bool)
    samples = src.shape[0]
    dims    = src.shape[1]
    
    # run-length compress by axis=1
    if axis==1:
        dst = src.detach().cpu().t().reshape(-1)
    else:
        dst = src.detach().cpu().reshape(-1)
    length = len(dst)
        
    if nz_pos is None:
        nz_pos = (dst!=0)
        
    non_zero_values = dst[nz_pos].to(torch.float16)
    
    #######################
    # Extract change points
    #######################
    nz_pos = nz_pos.char()
    ind = torch.arange(0, length)#, dtype=torch.int16)
        
    # none zero position
    nz_cp = nz_pos - torch.cat((torch.CharTensor([0]), nz_pos[0:-1]), 0)
    nz_head = ind[nz_cp==1]
    nz_tail = ind[nz_cp==-1]
    
    if get_position==0:
        # Fully encode
        info = {
            'samples': samples, #int
            'dims': dims, #int
            'non_zero_values': non_zero_values, # float16, tensor, 1d-array
            'nz_head': nz_head.to(torch.int16), # int16, tensor, 1d-array
            'nz_tail': nz_tail.to(torch.int16), # int16, tensor, 1d-array
        }
        return info
    elif get_position==1:
        # Without position
        info = {
            'samples': samples, #int
            'dims': dims, #int
            'non_zero_values': non_zero_values, # float16, tensor, 1d-array
        }
        return info
    elif get_position==2:
        # Duplicate position
        info = {
            'samples': samples, #int
            'dims': dims, #int
            'non_zero_values': non_zero_values, # float16, tensor, 1d-array
            'nz_head': nz_head.to(torch.int16), # int16, tensor, 1d-array
            'nz_tail': nz_tail.to(torch.int16), # int16, tensor, 1d-array
        }
        position = {
            'nz_head': nz_head, # long, tensor, cpu, 1d-array
            'nz_tail': nz_tail, # long, tensor, cpu, 1d-array
        }
        return info, position



def sparse_decode16(info, device, set_position=None, axis=1):
    # Extract data
    samples = info['samples']
    dims    = info['dims']
    length  = samples * dims # info['length']
    if set_position is None:
        nz_head = info['nz_head'].long()
        nz_tail = info['nz_tail'].long()
    else:
        nz_head = set_position['nz_head']#.long()
        nz_tail = set_position['nz_tail']#.long()
    
    nz_cp = torch.zeros(length, dtype=torch.int8)
    
    nz_cp[nz_head] = 1
    nz_cp[nz_tail] = -1
    
    nz_pos = torch.cumsum(nz_cp, dim=0).bool()
    
    dst = torch.zeros(length)
    dst[nz_pos] = info['non_zero_values'].float()
    
    if axis==1:
        dst = dst.reshape((dims, samples)).t().to(device)
    else:
        dst = dst.reshape((samples, dims)).to(device)

    
    return dst, nz_pos



# encode
def encode_grad_topk(ge):
    top_k = args.top_k
    x = ge.clone()
    
    sh = x.shape
    x = x.flatten()
    prob = x.abs()
    values, indices = prob.sort()
    sv = x
    qa0 = indices[-top_k:].int()
    qa1 = sv[indices[-top_k:]]
    
    info = {
        'sh': sh,
        'qa0': qa0.detach().cpu().int(),    # int32
        'qa1': qa1.detach().cpu().float(),  # float32
    }
    
    return info

def decode_grad_topk(c):
    sh = c['sh']
    l = sh[0] * sh[1]
    qa0 = c['qa0'].long().to(device)
    qa1 = c['qa1'].float().to(device)
    ge = torch.zeros(l).to(device)
    ge[qa0] = qa1
    return ge.reshape(sh[0], sh[1])

def encode_grad_topk16(ge):
    top_k = args.top_k
    x = ge.clone()
    
    sh = x.shape
    x = x.flatten()
    prob = x.abs()
    values, indices = prob.sort()
    sv = x
    qa0 = indices[-top_k:].int()
    qa1 = sv[indices[-top_k:]]
    info = {
        'sh': sh, # shape
        'qa0': qa0.detach().cpu().to(torch.int16),    # int16, index, 1d-array
        'qa1': qa1.detach().cpu().to(torch.float16),  # float16, values, 1d-array
    }
    
    return info

def decode_grad_topk16(c):
    sh = c['sh']
    l = sh[0] * sh[1]
    qa0 = c['qa0'].long().to(device) # index
    qa1 = c['qa1'].float().to(device) # values
    ge = torch.zeros(l).to(device)
    ge[qa0] = qa1
    return ge.reshape(sh[0], sh[1])


def reduce_encode_emb(emb, cid, header='tr'):
    pos = None
    theoretical_size = 0
    if args.reduction=='svfl16':
        ec, pos = sparse_encode16(emb, device, get_position=2, axis=args.run_axis)
        # bit
        q0 = math.ceil(math.log2(ec['samples']))
        q1 = math.ceil(math.log2(ec['dims']))
        q2 = math.ceil(math.log2(ec['samples'] * ec['dims']))
        theoretical_size = ec['non_zero_values'].shape[0] * 16 + (ec['nz_head'].shape[0] + ec['nz_tail'].shape[0]) * q2 + q0 + q1
    elif args.reduction=='org':
        ec = emb.detach().cpu()
        theoretical_size = (emb.shape[0] * emb.shape[1]) * 32
    elif args.reduction=='int':
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
    elif args.reduction=='topk':
        ec, pos = sparse_encode(emb, device, get_position=2, axis=args.run_axis)
        q0 = math.ceil(math.log2(ec['samples']))
        q1 = math.ceil(math.log2(ec['dims']))
        q2 = math.ceil(math.log2(ec['samples'] * ec['dims']))
        theoretical_size = ec['non_zero_values'].shape[0] * 32 + (ec['nz_head'].shape[0] + ec['nz_tail'].shape[0]) * q2 + q0 + q1
    elif args.reduction=='topk16':
        ec, pos = sparse_encode16(emb, device, get_position=2, axis=args.run_axis)
        q0 = math.ceil(math.log2(ec['samples']))
        q1 = math.ceil(math.log2(ec['dims']))
        q2 = math.ceil(math.log2(ec['samples'] * ec['dims']))
        theoretical_size = ec['non_zero_values'].shape[0] * 16 + (ec['nz_head'].shape[0] + ec['nz_tail'].shape[0]) * q2 + q0 + q1
    elif args.reduction=='svfl':
        ec, pos = sparse_encode(emb, device, get_position=2, axis=args.run_axis)
        q0 = math.ceil(math.log2(ec['samples']))
        q1 = math.ceil(math.log2(ec['dims']))
        q2 = math.ceil(math.log2(ec['samples'] * ec['dims']))
        theoretical_size = ec['non_zero_values'].shape[0] * 32 + (ec['nz_head'].shape[0] + ec['nz_tail'].shape[0]) * q2 + q0 + q1
    else:
        print('error')
        
    torch.save(ec, f'{comm_dir}/{header}_emb_{cid}.pt')
    theoretical_size = math.ceil(theoretical_size/8.0) # bit to byte
    return pos, theoretical_size

def reduce_decode_emb(cid, header='tr'):
    e = torch.load(f'{comm_dir}/{header}_emb_{cid}.pt')
    nz_pos = None
    if args.reduction=='svfl16':
        e, nz_pos = sparse_decode16(e, device, axis=args.run_axis)
    elif args.reduction=='org':
        e = e.to(device)
    elif args.reduction=='int':
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
    elif args.reduction=='topk':
        e, nz_pos = sparse_decode(e, device, axis=args.run_axis)
    elif args.reduction=='topk16':
        e, nz_pos = sparse_decode16(e, device, axis=args.run_axis)
    elif args.reduction=='svfl':
        e, nz_pos = sparse_decode(e, device, axis=args.run_axis)
    else:
        print('error')
        e = None

    return e, nz_pos


def reduce_encode_grad(ge, nz_pos, cid):
    theoretical_size = 0
    if args.reduction=='svfl16':
        gec = sparse_encode16(ge, device, get_position=1, nz_pos=nz_pos, axis=args.run_axis) # without position
        q0 = math.ceil(math.log2(gec['samples']))
        q1 = math.ceil(math.log2(gec['dims']))
        # NO NEED INDICES
        theoretical_size = gec['non_zero_values'].shape[0] * 16 + q0 + q1
    elif args.reduction=='org':
        gec = ge.detach().cpu()
        theoretical_size = ge.shape[0] * ge.shape[1] * 32
    elif args.reduction=='int':
        gec = ge.detach().cpu()
        theoretical_size = ge.shape[0] * ge.shape[1] * 32
    elif args.reduction=='q4':
        mi = ge.min()
        ma = ge.max()
        gec = {'mi': mi.detach().cpu(), 'ma': ma.detach().cpu() ,'ge':((ge-mi)/(ma-mi)*16).to(torch.quint4x2).detach().cpu()}
        theoretical_size = ge.shape[0] * ge.shape[1] * 4
    elif args.reduction=='q8':
        mi = ge.min()
        ma = ge.max()
        gec = {'mi': mi.detach().cpu(), 'ma': ma.detach().cpu(), 'ge':((ge-mi)/(ma-mi)*255).byte().detach().cpu()}
        theoretical_size = ge.shape[0] * ge.shape[1] * 8
    elif args.reduction=='q16':
        gec = ge.to(torch.float16).detach().cpu()
        theoretical_size = ge.shape[0] * ge.shape[1] * 16
    elif args.reduction=='topk':
        gec = encode_grad_topk(ge)
        q0 = math.ceil(math.log2(gec['sh'][0]))
        q1 = math.ceil(math.log2(gec['sh'][1]))
        q2 = math.ceil(math.log2(gec['sh'][0] * gec['sh'][1]))
        theoretical_size = gec['qa0'].shape[0] * q2 + gec['qa1'].shape[0] * 32 + q0 + q1
    elif args.reduction=='topk16':
        gec = encode_grad_topk16(ge)
        q0 = math.ceil(math.log2(gec['sh'][0]))
        q1 = math.ceil(math.log2(gec['sh'][1]))
        q2 = math.ceil(math.log2(gec['sh'][0] * gec['sh'][1]))
        theoretical_size = gec['qa0'].shape[0] * q2 + gec['qa1'].shape[0] * 16 + q0 + q1
    elif args.reduction=='svfl':
        gec = sparse_encode(ge, device, get_position=1, nz_pos=nz_pos, axis=args.run_axis) # without position
        q0 = math.ceil(math.log2(gec['samples']))
        q1 = math.ceil(math.log2(gec['dims']))
        # NO NEED INDICES
        theoretical_size = gec['non_zero_values'].shape[0] * 32 + q0 + q1
    else:
        print('error')
        
    torch.save(gec, f'{comm_dir}/ge{cid}.pt')
    theoretical_size = math.ceil(theoretical_size/8.0) # bit to byte
    return theoretical_size
    
        
def reduce_decode_grad(cid, pos=None):
    ge = torch.load(f'{comm_dir}/ge{cid}.pt')
    if args.reduction=='svfl16':
        ge, _ = sparse_decode16(ge, device, set_position=pos, axis=args.run_axis)
    elif args.reduction=='org':
        ge = ge.to(device)
    elif args.reduction=='int':
        ge = ge.to(device)
    elif args.reduction=='q4':
        mi = ge['mi'].to(device)
        ma = ge['ma'].to(device)
        ge = ge['ge'].float().to(device)
        ge = ge * (ma-mi) / 16.0 + mi        
    elif args.reduction=='q8':
        mi = ge['mi'].to(device)
        ma = ge['ma'].to(device)
        ge = ge['ge'].float().to(device)
        ge = ge * (ma-mi) / 255.0 + mi
    elif args.reduction=='q16':
        ge = ge.float().to(device)
    elif args.reduction=='topk':
        ge = decode_grad_topk(ge)
    elif args.reduction=='topk16':
        ge = decode_grad_topk16(ge)
    elif args.reduction=='svfl':
        ge, _ = sparse_decode(ge, device, set_position=pos, axis=args.run_axis)
    else:
        print('error')
    return ge

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
layer_no_bias = args.layer_no_bias
sparse_embed_lambda = args.sparse_embed_lambda


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
class ClientModel(torch.nn.Module):
    def __init__(self, in_size, hidden_size):
        super(ClientModel, self).__init__()
        if layer_no_bias:
            self.i2h = torch.nn.Linear(in_size, hidden_size, bias=False)
            # Reproducability
            set_seed(seed)
            torch.nn.init.xavier_uniform_(self.i2h.weight.data)
        else:
            self.i2h = torch.nn.Linear(in_size, hidden_size, bias=True)
            # Reproducability
            set_seed(seed)
            torch.nn.init.xavier_uniform_(self.i2h.weight.data)
            torch.nn.init.ones_(self.i2h.bias.data)
            
                
    def forward(self, x):
        h = self.i2h(x)
        if args.client_activation == 'relu':
            h = F.relu(h)
        elif args.client_activation == 'leaky_relu':
            h = F.leaky_relu(h)
        elif args.client_activation == 'selu':
            h = F.selu(h)
        elif args.client_activation == 'elu':
            h = F.elu(h)
        elif args.client_activation == 'mish':
            h = F.mish(h)
        return h
    
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
# Client 1
###################
client_model_1 = ClientModel(len(tr_xcols1), dim_1)
client_model_1 = client_model_1.to(device)
if args.client_optimizer=='adam':
    optimizer_1 = torch.optim.Adam(client_model_1.parameters(), lr=lr)
elif args.client_optimizer=='sgd':
    optimizer_1 = torch.optim.SGD(client_model_1.parameters(), lr=lr)
elif args.client_optimizer=='rmsprop':
    optimizer_1 = torch.optim.RMSprop(client_model_1.parameters(), lr=lr)
    
###################
# Client 2
###################
client_model_2 = ClientModel(len(tr_xcols2), dim_2)
client_model_2 = client_model_2.to(device)
if args.client_optimizer=='adam':
    optimizer_2 = torch.optim.Adam(client_model_2.parameters(), lr=lr)
elif args.client_optimizer=='sgd':
    optimizer_2 = torch.optim.SGD(client_model_2.parameters(), lr=lr)
elif args.client_optimizer=='rmsprop':
    optimizer_2 = torch.optim.RMSprop(client_model_2.parameters(), lr=lr)

###################
# Client 3
###################
client_model_3 = ClientModel(len(tr_xcols3), dim_3)
client_model_3 = client_model_3.to(device)
if args.client_optimizer=='adam':
    optimizer_3 = torch.optim.Adam(client_model_3.parameters(), lr=lr)
elif args.client_optimizer=='sgd':
    optimizer_3 = torch.optim.SGD(client_model_3.parameters(), lr=lr)
elif args.client_optimizer=='rmsprop':
    optimizer_3 = torch.optim.RMSprop(client_model_3.parameters(), lr=lr)
    
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
    # Client 1
    ###################
    client_model_1.train()
    
    ###################
    # Client 2
    ###################
    client_model_2.train()
    
    ###################
    # Client 3
    ###################
    client_model_3.train()
    
    ###################
    # Server
    ###################
    server_model.train()
    
    epoch_count_send_1 = 0
    epoch_count_send_2 = 0
    epoch_count_send_3 = 0
    epoch_count_receive_1 = 0
    epoch_count_receive_2 = 0
    epoch_count_receive_3 = 0

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
    
    tr_loss = 0
    tr_pred = np.zeros(tr_y.shape)
    tr_true = np.zeros(tr_y.shape)
    
    
    ###################
    # Statistics
    ###################
    tr_sparse_embed_loss = 0
    tr_embed_sparsity = 0
    tr_grad_sparsity = 0
    

    for batch_index in tqdm(range(tr_batch_count)):
        ###################
        # Common
        ###################
        head = batch_index * batch_size
        tail = min(head + batch_size, tr_sample_count)
        si = tr_shuffle[head:tail]
        batch_uid = tr_uid[si]
        
        ###################
        # Client 1: Send
        ###################
        st = time.time()
        batch_x_1 = tr_x1[si,:].to(device)
        optimizer_1.zero_grad()
        emb_1 = client_model_1(batch_x_1)
        
        ## Data reduction for embed
        pos_1, size_e1t = reduce_encode_emb(emb_1, 1)
        total_time_1 += (time.time() - st)

        ###################
        # Client 2: Send
        ###################
        st = time.time()
        batch_x_2 = tr_x2[si,:].to(device)
        optimizer_2.zero_grad()
        emb_2 = client_model_2(batch_x_2)
        
        ## Data reduction for embed
        pos_2, size_e2t = reduce_encode_emb(emb_2, 2)
        total_time_2 += (time.time() - st)

        ###################
        # Client 3: Send
        ###################
        st = time.time()
        batch_x_3 = tr_x3[si,:].to(device)
        optimizer_3.zero_grad()
        emb_3 = client_model_3(batch_x_3)
        
        ## Data reduction for embed
        pos_3, size_e3t = reduce_encode_emb(emb_3, 3)
        total_time_3 += (time.time() - st)
            
        ###################
        # Server: Receive, Send
        ###################
        st = time.time()
        batch_y   = tr_y[si,:].to(device)
        optimizer.zero_grad()
        e1, nz_pos_1 = reduce_decode_emb(1)
        e2, nz_pos_2 = reduce_decode_emb(2)
        e3, nz_pos_3 = reduce_decode_emb(3)

        e1.requires_grad_(True) # Enable to get gradient
        e2.requires_grad_(True) # Enable to get gradient
        e3.requires_grad_(True) # Enable to get gradient
        emb = torch.cat((e1, e2, e3), 1)
        
        pred_y = server_model(emb)
        loss = criterion(pred_y, batch_y)
            
        ## L1-norm
        if sparse_embed_lambda > 0:
            if args.norm == 1:
                sparse_embed_loss_1 = torch.norm(e1, p=1, dim=1).mean() # L1 norm of each sample, mean over samples in 1 batch
                sparse_embed_loss_2 = torch.norm(e2, p=1, dim=1).mean() # L1 norm of each sample, mean over samples in 1 batch
                sparse_embed_loss_3 = torch.norm(e3, p=1, dim=1).mean() # L1 norm of each sample, mean over samples in 1 batch
            else:
                sparse_embed_loss_1 = torch.norm(e1, p=2, dim=1).mean() # L2 norm of each sample, mean over samples in 1 batch
                sparse_embed_loss_2 = torch.norm(e2, p=2, dim=1).mean() # L2 norm of each sample, mean over samples in 1 batch
                sparse_embed_loss_3 = torch.norm(e3, p=2, dim=1).mean() # L2 norm of each sample, mean over samples in 1 batch
                
            loss = loss + sparse_embed_lambda * (sparse_embed_loss_1 + sparse_embed_loss_2 + sparse_embed_loss_3)/3.0

            
        loss.backward()
        optimizer.step()

        ## Gradient of embed
        ge1 = e1.grad 
        ge2 = e2.grad 
        ge3 = e3.grad
        
        size_g1t = reduce_encode_grad(ge1, nz_pos_1, 1)
        size_g2t = reduce_encode_grad(ge2, nz_pos_2, 2)
        size_g3t = reduce_encode_grad(ge3, nz_pos_3, 3)
            
        
        ## Training results
        tr_loss += loss.item()
        if sparse_embed_lambda > 0:
            tr_sparse_embed_loss += sparse_embed_loss_1.item()
            tr_sparse_embed_loss += sparse_embed_loss_2.item()
            tr_sparse_embed_loss += sparse_embed_loss_3.item()
        if args.loss=='bcewll':
            tr_pred[head:tail,:] = torch.sigmoid(pred_y).detach().cpu().numpy()
            tr_true[head:tail,:] = batch_y.detach().cpu().numpy()
        elif args.loss=='ce':
            tr_pred[head:tail,:] = torch.argmax(pred_y, dim=1, keepdim=True).detach().cpu().numpy()
            tr_true[head:tail,:] = batch_y.detach().cpu().numpy()
        else:
            print('error')

        total_time_s += (time.time() - st)
        
        ###################
        # Client 1: Receive
        ###################
        st = time.time()
        ge1 = reduce_decode_grad(1, pos=pos_1)
        emb_1.backward(ge1)
        optimizer_1.step()
        total_time_1 += (time.time() - st)
        
        ###################
        # Client 2: Receive
        ###################
        st = time.time()
        ge2 = reduce_decode_grad(2, pos=pos_2)
        emb_2.backward(ge2)
        optimizer_2.step()
        total_time_2 += (time.time() - st)
        
        ###################
        # Client 3: Receive
        ###################
        st = time.time()
        ge3 = reduce_decode_grad(3, pos=pos_3)
        emb_3.backward(ge3)
        optimizer_3.step()
        total_time_3 += (time.time() - st)

        
        ###################
        # Statistics
        ###################
        epoch_count_send_1 += 1
        epoch_count_send_2 += 1
        epoch_count_send_3 += 1
        epoch_count_receive_1 += 1
        epoch_count_receive_2 += 1
        epoch_count_receive_3 += 1

        # Practical size
        epoch_size_send_1    += os.path.getsize(f'{comm_dir}/tr_emb_1.pt')
        epoch_size_send_2    += os.path.getsize(f'{comm_dir}/tr_emb_2.pt')
        epoch_size_send_3    += os.path.getsize(f'{comm_dir}/tr_emb_3.pt')
        epoch_size_receive_1 += os.path.getsize(f'{comm_dir}/ge1.pt')
        epoch_size_receive_2 += os.path.getsize(f'{comm_dir}/ge2.pt')
        epoch_size_receive_3 += os.path.getsize(f'{comm_dir}/ge3.pt')
        
        # Theoretical size
        epoch_size_send_1t    += size_e1t
        epoch_size_send_2t    += size_e2t 
        epoch_size_send_3t    += size_e3t
        epoch_size_receive_1t += size_g1t
        epoch_size_receive_2t += size_g2t
        epoch_size_receive_3t += size_g3t
        

        # Count sparsity
        tr_embed_sparsity += np.count_nonzero(np.abs(emb_1.detach().cpu().numpy()) <= sparse_embed_epsilon)
        tr_embed_sparsity += np.count_nonzero(np.abs(emb_2.detach().cpu().numpy()) <= sparse_embed_epsilon)
        tr_embed_sparsity += np.count_nonzero(np.abs(emb_3.detach().cpu().numpy()) <= sparse_embed_epsilon)
        
        tr_grad_sparsity  += np.count_nonzero(np.abs(ge1.detach().cpu().numpy()) <= sparse_grad_epsilon)
        tr_grad_sparsity  += np.count_nonzero(np.abs(ge2.detach().cpu().numpy()) <= sparse_grad_epsilon)
        tr_grad_sparsity  += np.count_nonzero(np.abs(ge3.detach().cpu().numpy()) <= sparse_grad_epsilon)
    
    
    ##########################################################################
    # Validation
    ##########################################################################
    print('Validation')
    
    ###################
    # Common
    ###################
    va_sample_count = len(va_uid)
    va_batch_count = math.ceil(va_sample_count / batch_size)
    
    ###################
    # Client 1
    ###################
    client_model_1.eval()
    
    ###################
    # Client 2
    ###################
    client_model_2.eval()
    
    ###################
    # Client 2
    ###################
    client_model_3.eval()

    ###################
    # Server
    ###################
    server_model.eval()
    va_loss = 0
    va_pred = np.zeros(va_y.shape)
    va_true = np.zeros(va_y.shape)
    
    
    va_embed_sparsity = 0
    
    
    for batch_index in tqdm(range(va_batch_count)):
        ###################
        # Common
        ###################
        head = batch_index * batch_size
        tail = min(head + batch_size, va_sample_count)
        batch_uid = va_uid[head:tail]
        
        ###################
        # Client 1: Send
        ###################
        st = time.time()
        batch_x_1 = va_x1[head:tail,:].to(device)
        emb_1 = client_model_1(batch_x_1)
        _, size_e1t = reduce_encode_emb(emb_1, 1, header='va')
        total_time_1 += (time.time() - st)
        
        ###################
        # Client 2
        ###################
        st = time.time()
        batch_x_2 = va_x2[head:tail,:].to(device)
        emb_2 = client_model_2(batch_x_2)
        _, size_e2t = reduce_encode_emb(emb_2, 2, header='va')
        total_time_2 += (time.time() - st)

        ###################
        # Client 3
        ###################
        st = time.time()
        batch_x_3 = va_x3[head:tail,:].to(device)
        emb_3 = client_model_3(batch_x_3)
        _, size_e3t = reduce_encode_emb(emb_3, 3, header='va')
        total_time_3 += (time.time() - st)

        ###################
        # Server: Receive, Send
        ###################
        st = time.time()
        batch_y   = va_y[head:tail,:].to(device)
        e1, _ = reduce_decode_emb(1, header='va')
        e2, _ = reduce_decode_emb(2, header='va')
        e3, _ = reduce_decode_emb(3, header='va')
        
        emb = torch.cat((e1, e2, e3), 1).to(device)
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

        ###################
        # Statistics
        ###################
        epoch_count_send_1 += 1
        epoch_count_send_2 += 1
        epoch_count_send_3 += 1

        # Practical size
        epoch_size_send_1 += os.path.getsize(f'{comm_dir}/va_emb_1.pt')
        epoch_size_send_2 += os.path.getsize(f'{comm_dir}/va_emb_2.pt')
        epoch_size_send_3 += os.path.getsize(f'{comm_dir}/va_emb_3.pt')
        # Theoretical size
        epoch_size_send_1t    += size_e1t
        epoch_size_send_2t    += size_e2t 
        epoch_size_send_3t    += size_e3t

        # Count sparsity
        va_embed_sparsity += np.count_nonzero(np.abs(emb_1.detach().cpu().numpy()) <= sparse_embed_epsilon)
        va_embed_sparsity += np.count_nonzero(np.abs(emb_2.detach().cpu().numpy()) <= sparse_embed_epsilon)
        va_embed_sparsity += np.count_nonzero(np.abs(emb_3.detach().cpu().numpy()) <= sparse_embed_epsilon)

        
    ###################
    # Statistics
    ###################
    total_count_send_1    += epoch_count_send_1
    total_count_send_2    += epoch_count_send_2
    total_count_send_3    += epoch_count_send_3
    total_count_receive_1 += epoch_count_receive_1
    total_count_receive_2 += epoch_count_receive_2
    total_count_receive_3 += epoch_count_receive_3
    
    total_size_send_1    += epoch_size_send_1
    total_size_send_2    += epoch_size_send_2
    total_size_send_3    += epoch_size_send_3
    total_size_receive_1 += epoch_size_receive_1
    total_size_receive_2 += epoch_size_receive_2
    total_size_receive_3 += epoch_size_receive_3

    total_size_send_1t    += epoch_size_send_1t
    total_size_send_2t    += epoch_size_send_2t
    total_size_send_3t    += epoch_size_send_3t
    total_size_receive_1t += epoch_size_receive_1t
    total_size_receive_2t += epoch_size_receive_2t
    total_size_receive_3t += epoch_size_receive_3t
    
    # total Loss
    tr_loss /= tr_batch_count
    va_loss /= va_batch_count
    
    # L1 embed loss (part of the total loss)
    tr_sparse_embed_loss /= tr_batch_count
        
    # Embed Sparsity
    tr_embed_sparsity /= (tr_sample_count * (dim_1 + dim_2 + dim_3))
    va_embed_sparsity /= (va_sample_count * (dim_1 + dim_2 + dim_3))
    # Grad sparsity
    tr_grad_sparsity /= (tr_sample_count * (dim_1 + dim_2 + dim_3))
    
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
    writer.add_scalar('stat_count/epoch_count_send_1',    epoch_count_send_1, epoch)
    writer.add_scalar('stat_count/epoch_count_send_2',    epoch_count_send_2, epoch)
    writer.add_scalar('stat_count/epoch_count_send_3',    epoch_count_send_3, epoch)
    writer.add_scalar('stat_count/epoch_count_receive_1', epoch_count_receive_1, epoch)
    writer.add_scalar('stat_count/epoch_count_receive_2', epoch_count_receive_2, epoch)
    writer.add_scalar('stat_count/epoch_count_receive_3', epoch_count_receive_3, epoch)
    writer.add_scalar('stat_count/epoch_count_all', epoch_count_send_1 + epoch_count_send_2 + epoch_count_send_3 + epoch_count_receive_1 + epoch_count_receive_2 + epoch_count_receive_3, epoch)
    
    writer.add_scalar('stat_size/epoch_size_send_1',     epoch_size_send_1, epoch)
    writer.add_scalar('stat_size/epoch_size_send_2',     epoch_size_send_2, epoch)
    writer.add_scalar('stat_size/epoch_size_send_3',     epoch_size_send_3, epoch)
    writer.add_scalar('stat_size/epoch_size_receive_1',  epoch_size_receive_1, epoch)
    writer.add_scalar('stat_size/epoch_size_receive_2',  epoch_size_receive_2, epoch)
    writer.add_scalar('stat_size/epoch_size_receive_3',  epoch_size_receive_3, epoch)
    writer.add_scalar('stat_size/epoch_size_all', epoch_size_send_1 + epoch_size_send_2 + epoch_size_send_3 + epoch_size_receive_1 + epoch_size_receive_2 + epoch_size_receive_3, epoch)

    writer.add_scalar('stat_size_t/epoch_size_send_1t',     epoch_size_send_1t, epoch)
    writer.add_scalar('stat_size_t/epoch_size_send_2t',     epoch_size_send_2t, epoch)
    writer.add_scalar('stat_size_t/epoch_size_send_3t',     epoch_size_send_3t, epoch)
    writer.add_scalar('stat_size_t/epoch_size_receive_1t',  epoch_size_receive_1t, epoch)
    writer.add_scalar('stat_size_t/epoch_size_receive_2t',  epoch_size_receive_2t, epoch)
    writer.add_scalar('stat_size_t/epoch_size_receive_3t',  epoch_size_receive_3t, epoch)
    writer.add_scalar('stat_size_t/epoch_size_allt', epoch_size_send_1t + epoch_size_send_2t + epoch_size_send_3t + epoch_size_receive_1t + epoch_size_receive_2t + epoch_size_receive_3t, epoch)

    writer.add_scalar('reg/tr_sparse_embed_loss_x_lambda', sparse_embed_lambda * tr_sparse_embed_loss, epoch) # with sparse_lambda
    writer.add_scalar('reg/tr_embed_sparsity', tr_embed_sparsity, epoch)
    writer.add_scalar('reg/va_embed_sparsity', va_embed_sparsity, epoch)
    writer.add_scalar('reg/tr_grad_sparsity',  tr_grad_sparsity, epoch)

    writer.flush()
    
    if va_score > best_va_score:
        best_va_score = va_score
        
        model_name = f'{model_dir}/{model_header}_server_model.pt'
        torch.save(server_model.state_dict(), model_name)
        patience_counter = 0
        
        model_name = f'{model_dir}/{model_header}_client_model_1.pt'
        torch.save(client_model_1.state_dict(), model_name)
        model_name = f'{model_dir}/{model_header}_client_model_2.pt'
        torch.save(client_model_2.state_dict(), model_name)
        model_name = f'{model_dir}/{model_header}_client_model_3.pt'
        torch.save(client_model_3.state_dict(), model_name)
        

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
# Statistics
###################
print(f'total_count_send_1: {total_count_send_1}')
print(f'total_count_send_2: {total_count_send_2}')
print(f'total_count_send_3: {total_count_send_3}')
print(f'total_count_receive_1: {total_count_receive_1}')
print(f'total_count_receive_2: {total_count_receive_2}')
print(f'total_count_receive_3: {total_count_receive_3}')
print('-'*10)
print(f'total_size_send_1: {total_size_send_1/2**30:.3f}')
print(f'total_size_send_2: {total_size_send_2/2**30:.3f}')
print(f'total_size_send_3: {total_size_send_3/2**30:.3f}')
print(f'total_size_receive_1: {total_size_receive_1/2**30:.3f}')
print(f'total_size_receive_2: {total_size_receive_2/2**30:.3f}')
print(f'total_size_receive_3: {total_size_receive_3/2**30:.3f}')
print('-'*10)
print(f'total_size_send_1t: {total_size_send_1t/2**30:.3f}')
print(f'total_size_send_2t: {total_size_send_2t/2**30:.3f}')
print(f'total_size_send_3t: {total_size_send_3t/2**30:.3f}')
print(f'total_size_receive_1t: {total_size_receive_1t/2**30:.3f}')
print(f'total_size_receive_2t: {total_size_receive_2t/2**30:.3f}')
print(f'total_size_receive_3t: {total_size_receive_3t/2**30:.3f}')
print('-'*10)
total_count_all = total_count_send_1 + total_count_send_2 + total_count_send_3 + total_count_receive_1 + total_count_receive_2  + total_count_receive_2
total_size_all   = total_size_send_1  + total_size_send_2  + total_size_send_3  + total_size_receive_1  + total_size_receive_2  + total_size_receive_3
total_size_allt  = total_size_send_1t + total_size_send_2t + total_size_send_3t + total_size_receive_1t + total_size_receive_2t + total_size_receive_3t
print(f'total_count_all: {total_count_all}')
print(f'total_size_all: {(total_size_all)/2**30:.3f} [GB]')
print(f'total_size_allt: {(total_size_allt)/2**30:.3f} [GB]')
print(f'bandwidth: {(total_size_all/total_count_all)/2**10:.3f} [KB/comm]')


writer.add_scalar('final/total_size_send_1', total_size_send_1/2**30)
writer.add_scalar('final/total_size_send_2', total_size_send_2/2**30)
writer.add_scalar('final/total_size_send_3', total_size_send_3/2**30)
writer.add_scalar('final/total_size_receive_1', total_size_receive_1/2**30)
writer.add_scalar('final/total_size_receive_2', total_size_receive_2/2**30)
writer.add_scalar('final/total_size_receive_3', total_size_receive_3/2**30)

writer.add_scalar('theoretical/total_size_send_1t', total_size_send_1t/2**30)
writer.add_scalar('theoretical/total_size_send_2t', total_size_send_2t/2**30)
writer.add_scalar('theoretical/total_size_send_3t', total_size_send_3t/2**30)
writer.add_scalar('theoretical/total_size_receive_1t', total_size_receive_1t/2**30)
writer.add_scalar('theoretical/total_size_receive_2t', total_size_receive_2t/2**30)
writer.add_scalar('theoretical/total_size_receive_3t', total_size_receive_3t/2**30)

writer.flush()




###################
# Best model
###################
print('Best model')
print(f'best_va_score: {best_va_score}')
print(f'{model_dir}/{model_header}_server_model.pt')
print(f'{model_dir}/{model_header}_client_model_1.pt')
print(f'{model_dir}/{model_header}_client_model_2.pt')
print(f'{model_dir}/{model_header}_client_model_3.pt')

# Test data
if args.data_format == 'paper':
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

elif args.data_format == 'aws':
    # Common
    te_uid = torch.LongTensor(np.load(f"{data_dir}/test/te_uid.npy", allow_pickle=False))
    
    # Client
    te_x1 = torch.FloatTensor(np.load(f"{data_dir}/test/te_x_1.npy", allow_pickle=False))
    te_x2 = torch.FloatTensor(np.load(f"{data_dir}/test/te_x_2.npy", allow_pickle=False))
    te_x3 = torch.FloatTensor(np.load(f"{data_dir}/test/te_x_3.npy", allow_pickle=False))
#     te_x4 = torch.FloatTensor(np.load(f"{data_dir}/test/te_x_4.npy", allow_pickle=False))
    te_xcols1 = np.load(f"{data_dir}/test/cols_1.npy", allow_pickle=False)
    te_xcols2 = np.load(f"{data_dir}/test/cols_2.npy", allow_pickle=False)
    te_xcols3 = np.load(f"{data_dir}/test/cols_3.npy", allow_pickle=False)
#     te_xcols4 = np.load(f"{data_dir}/test/cols_4.npy", allow_pickle=False)

    # Server
    te_y = torch.Tensor(np.load(f"{data_dir}/test/te_y.npy", allow_pickle=False))

###################
# Common
###################
te_sample_count = len(te_uid)
te_batch_count = math.ceil(te_sample_count / batch_size)

###################
# Client 1
###################
client_model_1.load_state_dict(torch.load(f'{model_dir}/{model_header}_client_model_1.pt'))
client_model_1 = client_model_1.to(device)
client_model_1.eval()

###################
# Client 2
###################
client_model_2.load_state_dict(torch.load(f'{model_dir}/{model_header}_client_model_2.pt'))
client_model_2 = client_model_2.to(device)
client_model_2.eval()

###################
# Client 3
###################
client_model_3.load_state_dict(torch.load(f'{model_dir}/{model_header}_client_model_3.pt'))
client_model_3 = client_model_3.to(device)
client_model_3.eval()

###################
# Server
###################
server_model.load_state_dict(torch.load(f'{model_dir}/{model_header}_server_model.pt'))
server_model = server_model.to(device)
server_model.eval()
te_pred = np.zeros(te_y.shape)
te_true = np.zeros(te_y.shape)

###################
# Statistics
###################
te_embed_sparsity = 0


for batch_index in tqdm(range(te_batch_count)):
    ###################
    # Common
    ###################
    head = batch_index * batch_size
    tail = min(head + batch_size, te_sample_count)        
    batch_uid = te_uid[head:tail]

    ###################
    # Client 1: Send
    ###################
    batch_x_1 = te_x1[head:tail,:].to(device)
    emb_1 = client_model_1(batch_x_1)
    reduce_encode_emb(emb_1, 1, header='te')

    ###################
    # Client 2: Send
    ###################
    batch_x_2 = te_x2[head:tail,:].to(device)
    emb_2 = client_model_2(batch_x_2)
    reduce_encode_emb(emb_2, 2, header='te')

    ###################
    # Client 3: Send
    ###################
    batch_x_3 = te_x3[head:tail,:].to(device)
    emb_3 = client_model_3(batch_x_3)
    reduce_encode_emb(emb_3, 3, header='te')

    ###################
    # Server: Receive
    ###################
    batch_y   = te_y[head:tail,:].to(device) # Label
    e1, _ = reduce_decode_emb(1, header='te')
    e2, _ = reduce_decode_emb(2, header='te')
    e3, _ = reduce_decode_emb(3, header='te')
    emb = torch.cat((e1, e2, e3), 1).to(device) # Concat horizontally
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
    te_embed_sparsity += np.count_nonzero(np.abs(emb_1.detach().cpu().numpy()) <= sparse_embed_epsilon)
    te_embed_sparsity += np.count_nonzero(np.abs(emb_2.detach().cpu().numpy()) <= sparse_embed_epsilon)
    te_embed_sparsity += np.count_nonzero(np.abs(emb_3.detach().cpu().numpy()) <= sparse_embed_epsilon)

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

te_embed_sparsity /= (te_sample_count * (dim_1 + dim_2 + dim_3))
print(f'te_sparsity: {te_embed_sparsity:.6f}')

writer.add_scalar('final/te_embed_sparsity',  te_embed_sparsity)
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
    'layer_no_bias': [args.layer_no_bias],
    'log_comment': [args.log_comment],
    'loss': [args.loss],
    'lr': [args.lr],
    'model_dir': [args.model_dir],
    'model_header': [args.model_header],
    'n_epochs': [args.n_epochs],
    'out_size': [args.out_size],
    'patience': [args.patience],
    'reduction': [args.reduction],
    'seed': [args.seed],
    'sparse_embed_epsilon': [args.sparse_embed_epsilon],
    'sparse_embed_lambda': [args.sparse_embed_lambda],
    'sparse_grad_epsilon': [args.sparse_grad_epsilon],
    'top_k': [args.top_k],
    
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

    
    'total_time_1': [total_time_1],
    'total_time_2': [total_time_2],
    'total_time_3': [total_time_3],
    'total_time_s': [total_time_s],
    'te_score': [te_score],
    'te_embed_sparsity': [te_embed_sparsity],
    'time': [end_time-start_time],
})

if os.path.exists(args.tsv_name):
    result_all = pd.read_csv(args.tsv_name, delimiter='\t')
    result_all = result_all.append(result)
    result_all.to_csv(args.tsv_name, sep='\t', index=False)
else:
    result.to_csv(args.tsv_name, sep='\t', index=False)


