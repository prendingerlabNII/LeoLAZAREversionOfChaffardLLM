import numpy as np
import torch
import torch.nn as nn
from torch import optim
from transformers import LlamaConfig, LlamaModel, LlamaTokenizer
from einops import rearrange
from models.PatchTST import AttentionLayer, FullAttention
from embed import PositionalEmbedding
import sys

class Llama(nn.Module):
    def __init__(self, configs, device):
        super(Llama, self).__init__()
        self.isllama = configs.isllama
        self.voc = configs.voc
        self.patch_size = configs.patch_size
        self.pretrain = configs.pretrain
        self.stride = configs.stride
        self.patch_num = (configs.seq_len - self.patch_size) // self.stride + 1
        self.d_model = configs.d_model
        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride)) 
        self.patch_num += 1
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.pos_embed = PositionalEmbedding(d_model=5120)
        if configs.isllama :
            if configs.pretrain :
                self.llama2 = LlamaModel.from_pretrained('meta-llama/Llama-2-13b-hf', token='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',output_attentions=True, output_hidden_states=True)
        self.in_layer = nn.Linear(configs.patch_size, self.d_model)

        # self.token = LlamaTokenizer.from_pretrained('meta-llama/Llama-2-13b-hf', token='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')
        self.in_layer_voc = nn.Linear(in_features= 32000,out_features= self.voc)
        self.attention = AttentionLayer(FullAttention(False, 5, attention_dropout=configs.dropout,output_attention= False,configs=configs), configs.d_model, configs.n_heads)
        self.out_layer = nn.Linear(configs.d_model * self.patch_num, configs.pred_len)
        # self.llama2 = self.llama2[:configs.gpt_layers]
        embed = self.llama2.get_input_embeddings()
        self.voc_in = embed(torch.tensor(range(32000)))
        self.voc_in=self.voc_in.to(device = device)
        # self.prompt = f'This dataset is the Bitcoin daily price chart. Each data point indicates the percent change in the opening, high, low and closing price of Bitcoin as well as the volatility and 2 exponential moving averages. Below is the information about the input time series: [Domain]: Bitcoin is the leading cryptocurrency, the bitcoin price is a highly volatile price chart which is globally on an upward trend. The chart oscillates between bull and bear markets which last about a year. [Instructions]: Predict the next {self.seq_len} steps given the previous {self.pred_len} steps information attached.'

        # self.prompt = self.token.encode(self.prompt, return_tensors = 'pt')
        # self.prompt = self.embed(self.prompt).to(device)
        

        if configs.freeze and configs.pretrain:
            for i, (name, param) in enumerate(self.llama2.named_parameters()):
                if 'layernorm' in name or 'norm' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        self.llama2.layers = self.llama2.layers[:configs.gpt_layers]
        for layer in (self.llama2, self.in_layer, self.out_layer, self.in_layer_voc, self.attention, self.pos_embed):
            layer.float().to(device=device)
            layer.train()
        
        self.cnt = 0

    def forward(self, x, itr):
        B, L, M = x.shape
        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False)+ 1e-5).detach() 
        x /= stdev
        
       
        x = rearrange(x, 'b l m -> b m l')
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_size, step=self.stride)
        x = rearrange(x, 'b m n p -> (b m) n p')
        # print(self.in_layer(x))
        outputs = self.in_layer(x)
        # print(outputs.shape)

        text_prot = self.in_layer_voc(self.voc_in.permute(1, 0)).permute(1,0)
        text_prot = torch.broadcast_to(text_prot, (x.shape[0],-1,-1)).detach()
        outputs, attns = self.attention(outputs, text_prot, text_prot, None , None)

        outputs = outputs + self.pos_embed(outputs)
        # outputs = torch.cat((self.prompt, outputs), dim = 0)
        if self.isllama:
            outputs = self.llama2(inputs_embeds=outputs).last_hidden_state
        outputs = self.out_layer(outputs.reshape(B*M, -1))
        outputs = rearrange(outputs, '(b m) l -> b l m', b=B)
        outputs = outputs * stdev
        outputs = outputs + means
        return outputs


