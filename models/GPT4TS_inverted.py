import numpy as np
import torch
import torch.nn as nn
from torch import optim

from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from transformers import BertTokenizer, BertModel
from einops import rearrange
from embed import DataEmbedding, DataEmbedding_wo_time, PositionalEmbedding
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from models.PatchTST import AttentionLayer, FullAttention
from sklearn.decomposition import PCA

class GPT4TSi(nn.Module):
    
    def __init__(self, configs, device):
        super(GPT4TSi, self).__init__()
        self.is_gpt = configs.is_gpt
        self.patch_size = configs.patch_size
        self.pretrain = configs.pretrain
        self.stride = configs.stride
        self.patch_num = (configs.seq_len - self.patch_size) // self.stride + 1
        self.attention = AttentionLayer(FullAttention(False, 5, attention_dropout=configs.dropout,output_attention= False,configs=configs), configs.d_model, configs.n_heads)
        self.padding_patch_layer = nn.ReplicationPad1d((self.stride, 0)) 
        self.patch_num += 1
        # self.in_layer_voc = nn.Linear(in_features= 32000,out_features= self.voc)

        if configs.is_gpt:
            if configs.pretrain:
                self.gpt2 = GPT2Model.from_pretrained('gpt2', output_attentions=True, output_hidden_states=True)  # loads a pretrained GPT-2 base model
            else:
                print("------------------no pretrain------------------")
                self.gpt2 = GPT2Model(GPT2Config())
            self.gpt2.h = self.gpt2.h[:configs.gpt_layers]
            print("gpt2 = {}".format(self.gpt2))
        pca = PCA()
        self.in_layer = nn.Linear(configs.seq_len, configs.d_model)
        self.in_layer1 = nn.Linear(configs.patch_size, configs.d_model)
        self.pos_embed = PositionalEmbedding(d_model=768)
        self.out_layer = nn.Linear(configs.d_model, configs.pred_len, bias = True)
        embed = self.gpt2.get_input_embeddings()
        self.voc_in = embed(torch.tensor(range(50257)))
        self.voc_in = self.voc_in.to(device)
        self.in_layer_voc = nn.Linear(in_features= 50257,out_features= configs.voc)

        if configs.freeze and configs.pretrain:
            for i, (name, param) in enumerate(self.gpt2.named_parameters()):
                if 'ln' in name or 'wpe' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        for layer in (self.gpt2, self.in_layer,self.in_layer1, self.out_layer, self.pos_embed, self.attention, self.in_layer_voc):
            layer.to(device=device)
            layer.train()
        
        self.cnt = 0


    def forward(self, x, itr):
        B, L, M = x.shape
        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False)+ 1e-5).detach() 
        x /= stdev

        x = rearrange(x, 'b l m -> b m l')
        # x = rearrange(x, 'b m l -> b m l')

        outputs = self.in_layer(x)
        if self.is_gpt:
            outputs = self.gpt2(inputs_embeds=outputs).last_hidden_state

        outputs = self.out_layer(outputs)
        outputs = rearrange(outputs, 'b m l -> b l m')

        outputs = outputs * stdev
        outputs = outputs + means

        return outputs
