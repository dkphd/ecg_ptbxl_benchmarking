import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from fastai.layers import Flatten, Lambda
# from fastai.core import *

from typing import Iterable, Optional

def listify(p=None, q=None):
    "Make `p` listy and the same length as `q`."
    if p is None: p=[]
    elif isinstance(p, str):          p = [p]
    elif not isinstance(p, Iterable): p = [p]
    #Rank 0 tensors in PyTorch are Iterable but don't have a length.
    else:
        try: a = len(p)
        except: p = [p]
    n = q if type(q)==int else len(p) if q is None else len(q)
    if len(p)==1: p = p * n
    assert len(p)==n, f'List len mismatch ({len(p)} vs {n})'
    return list(p)


def bn_drop_lin(n_in:int, n_out:int, bn:bool=True, p:float=0., actn:Optional[nn.Module]=None):
    "Sequence of batchnorm (if `bn`), dropout (with `p`) and linear (`n_in`,`n_out`) layers followed by `actn`."
    layers = [nn.BatchNorm1d(n_in)] if bn else []
    if p != 0: layers.append(nn.Dropout(p))
    layers.append(nn.Linear(n_in, n_out))
    if actn is not None: layers.append(actn)
    return layers

class AdaptiveConcatPoolRNN(nn.Module):
    def __init__(self, bidirectional):
        super().__init__()
        self.bidirectional = bidirectional
    def forward(self,x):
        #input shape bs, ch, ts
        t1 = nn.AdaptiveAvgPool1d(1)(x)
        t2 = nn.AdaptiveMaxPool1d(1)(x)
        
        if(self.bidirectional is False):
            t3 = x[:,:,-1]
        else:
            channels = x.size()[1]
            t3 = torch.cat([x[:,:channels,-1],x[:,channels:,0]],1)
        out=torch.cat([t1.squeeze(-1),t2.squeeze(-1),t3],1) #output shape bs, 3*ch
        return out

class RNN1d(nn.Sequential):
    def __init__(self,input_channels,num_classes,lstm=True,hidden_dim=256,num_layers=2,bidirectional=False,ps_head=0.5,act_head="relu",lin_ftrs_head=None,bn=True):
        layers_tmp = []
        #bs, ch, ts -> ts, bs, ch
        #LSTM
        layers_tmp.append(Lambda(lambda x: x.transpose(1,2)))
        layers_tmp.append(Lambda(lambda x: x.transpose(0,1)))
        if(lstm):
            layers_tmp.append(nn.LSTM(input_size=input_channels,hidden_size=hidden_dim,num_layers=num_layers,bidirectional=bidirectional))
        else:
            layers_tmp.append(nn.GRU(input_size=input_channels,hidden_size=hidden_dim,num_layers=num_layers,bidirectional=bidirectional))
        #pooling
        layers_tmp.append(Lambda(lambda x: x[0].transpose(0,1)))
        layers_tmp.append(Lambda(lambda x: x.transpose(1,2)))

        layers_head =[]
        
        layers_head.append(AdaptiveConcatPoolRNN(bidirectional))

        #classifier
        nf = 3*hidden_dim if bidirectional is False else 6*hidden_dim
        lin_ftrs_head = [nf, num_classes] if lin_ftrs_head is None else [nf] + lin_ftrs_head + [num_classes]
        ps_head = listify(ps_head)
        if len(ps_head)==1: 
            ps_head = [ps_head[0]/2] * (len(lin_ftrs_head)-2) + ps_head
        actns = [nn.ReLU(inplace=True) if act_head=="relu" else nn.ELU(inplace=True)] * (len(lin_ftrs_head)-2) + [None]
    
        for ni,no,p,actn in zip(lin_ftrs_head[:-1],lin_ftrs_head[1:],ps_head,actns):
            layers_head+=bn_drop_lin(ni,no,bn,p,actn)
        layers_head=nn.Sequential(*layers_head)
        layers_tmp.append(layers_head)

        super().__init__(*layers_tmp)
    
    def get_layer_groups(self):
        return (self[-1],)
    
    def get_output_layer(self):
        return self[-1][-1]
    
    def set_output_layer(self,x):
        self[-1][-1] = x
