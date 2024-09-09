import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
from transformers import LlamaConfig, LlamaModel, LlamaTokenizer, AutoModel, AutoTokenizer
from einops import rearrange
from models.PatchTST import AttentionLayer, FullAttention

from embed import PositionalEmbedding
import sys
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from utils.tools import sup_res_lines, fit_trendlines_single, fit_trendlines_high_low
from utils.srlinesnew import parallel_string
from models.RevIN import RevIN
from math import sqrt

class Llamatest(nn.Module):
    def __init__(self, configs, device):
        # TOKENIZER_PARRALLELISM = True
        super(Llamatest, self).__init__()
        self.isllama = configs.isllama
        self.voc = configs.voc
        self.batch_size = configs.batch_size
        self.patch_size = configs.patch_size
        self.pretrain = configs.pretrain
        self.stride = configs.stride
        self.patch_num = (configs.seq_len - self.patch_size) // self.stride + 1
        self.d_model = configs.d_model
        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride)) 
        self.patch_num += 1
        self.d_ff = configs.d_ff
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.pos_embed = PositionalEmbedding(d_model=self.d_model)
        self.head_nf = self.d_ff
        self.reprogramming_layer = ReprogrammingLayer(configs.d_model, configs.n_heads, self.d_ff, configs.d_model)
        self.num_features = configs.num_features
        self.output_projection = FlattenHead(configs.enc_in, self.head_nf, self.pred_len,
                                                 head_dropout=configs.dropout)    
        
        self.llama2 = LlamaModel.from_pretrained('meta-llama/Llama-2-13b-hf', token='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',output_attentions=True, output_hidden_states=True, num_hidden_layers = configs.gpt_layers)
        # self.llama2 = AutoModel.from_pretrained("01-ai/Yi-6B", trust_remote_code=True)
        self.in_layer = nn.Linear(configs.seq_len, self.d_model)
        self.in_layer1 = nn.Linear(7, self.d_model)
        # self.in_layer2 = nn.Linear(4, self.d_model)

        self.token = LlamaTokenizer.from_pretrained('meta-llama/Llama-2-13b-hf', token='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')
        # self.token = AutoTokenizer.from_pretrained("01-ai/Yi-6B", trust_remote_code=True)

        self.revin_layer = RevIN(num_features=self.num_features, affine = True, subtract_last = True)

        self.device = device
        self.in_layer_voc = nn.Linear(in_features= 32000,out_features= self.voc)
        # self.attention = AttentionLayer(FullAttention(False, 5, attention_dropout=configs.dropout,output_attention= False,configs=configs), configs.d_model, configs.n_heads)
        self.out_layer = nn.Linear(configs.d_model, self.pred_len)

        self.embed = self.llama2.get_input_embeddings()
        self.voc_in = self.embed(torch.tensor(range(32000)))
        self.pca = configs.pca
        if self.pca == 1 :
            pca = PCA(n_components=configs.voc)
            self.voc_in = self.voc_in.detach().cpu().numpy()
            pca.fit_transform(X = self.voc_in)
            self.voc_in = torch.tensor(pca.components_)
        elif self.pca == 2 :
            kme = KMeans(n_clusters=configs.voc)
            self.voc_in = self.voc_in.detach().numpy()
            kme.fit(self.voc_in)
            self.voc_in = torch.tensor(kme.cluster_centers_)

        self.voc_in=self.voc_in.to(device = device)

    
        for i, (name, param) in enumerate(self.llama2.named_parameters()):
            if 'layernorm' in name or 'norm' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        
        # self.llama2.layers = self.llama2.layers[:configs.gpt_layers]

        self.llama2.to(device, dtype = torch.bfloat16)
                
        # print(self.llama2.layers)
        # self.llama2.train()

        for layer in (self.in_layer,self.in_layer1, self.out_layer, self.in_layer_voc, self.pos_embed, self.revin_layer, self.reprogramming_layer, self.output_projection):
            layer.float().to(device=device)
            layer.train()
        
        self.cnt = 0

    def forward(self, x, itr):
        close = x[:,:,-1]
        high = x[:,:,-3]
        low = x[:,:, -2]
        x = x[:,:,:]
        maxprice = torch.max(close[:,:])
        minprice = torch.min(close[:,:])
        meanval = torch.mean(close[0,:])
        # SRlines = sup_res_lines(x[0,:,-1].cpu().detach().numpy())

        B, L, M = x.shape
        
        # df = pd.DataFrame(x[0, :, :].cpu().detach().numpy())
        # df = df.rename(columns={5:"open",6:"high",7:"low",8:"close"}) #### Attention to this line

        # SR_prompt = parallel_string(df, self.seq_len)
        
        x = self.revin_layer(x, 'norm')

        x = rearrange(x, 'b l m -> b m l')
        # x2 = x[:,:,-2:]
        # out1 = self.in_layer1(x1)
        # out2 = self.in_layer2(x2)
        # print(self.in_layer(x))
        outputs = self.in_layer(x)
        # print(outputs.shape)
        # print(lastvol)
        # print(lastRSI)
        # print(lastprice)

        prompt = f'This dataset is the Bitcoin daily price chart.'\
            f'Below is the information about the input time series:' \
            f'[Domain]: Bitcoin is the leading cryptocurrency. The bitcoin price is a highly volatile price chart which is globally on an upward trend although it oscillates between bull and bear market cycles that last around 1 to 2 years.' \
            f'Each data point indicates the opening, high, low and closing price of Bitcoin as well as the volume.' \
            f'[Instructions]: Predict the data for the next {self.pred_len} steps given the previous {self.seq_len} steps.'\
            f'[Statistics]: The input has a minimum value of {minprice:.0f} and a maximum value of {maxprice:.0f}, with an average value of {meanval:.0f}.'\
            f'Your predictions should take into account the behavior that Bitcoin prices tend to revert when approaching these support and resistance levels.'\
            # f'{SR_prompt}'

        # print(prompt)

            # f'HERE THE PRICE HITS A SUPPORT LINE AND IS EXPECTED TO REVERT, THE PRICE SHOULD BOUNCE STRONGLY UPWARDS.' \

            # f'The three most statistically prominent price levels are {float(SRlines[0]):.0f}, {float(SRlines[1]):.0f} and {float(SRlines[2]):.0f}.'
        

        prompt = self.token.encode(prompt, return_tensors = 'pt').to(self.device)
        prompt = self.embed(prompt)
        
        if self.pca == 0 :
            text_prot = self.in_layer_voc(self.voc_in.permute(1, 0)).permute(1,0).detach()
        elif self.pca == 1 or self.pca == 2 :
            text_prot = self.voc_in

        outputs = self.reprogramming_layer(outputs, text_prot, text_prot)
        # print(outputs.shape)
        prompt = torch.broadcast_to(prompt, (outputs.shape[0],-1,-1)).detach()
        outputs = torch.cat((prompt, outputs), dim = 1)
        outputs = outputs
        with torch.autocast(device_type= "cuda", dtype = torch.bfloat16):
            outputs = self.llama2(inputs_embeds=outputs).last_hidden_state

        dec_out = outputs[:,:,:self.d_ff]
        dec_out = self.output_projection(dec_out[:,-self.num_features:,:])
       

        # outputs = self.out_layer(outputs)
        # outputs = outputs[:,-12:,:]
        # outputs = rearrange(outputs, 'b m l -> b l m')
        dec_out = rearrange(dec_out, 'b m l -> b l m')

        # outputs = self.revin_layer(outputs, 'denorm')
        dec_out = self.revin_layer(dec_out, 'denorm')
        
        


        return dec_out


class ReprogrammingLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_keys=None, d_llm=None, attention_dropout=0.1):
        super(ReprogrammingLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)

        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.value_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.out_projection = nn.Linear(d_keys * n_heads, d_llm)
        self.n_heads = n_heads
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, target_embedding, source_embedding, value_embedding):
        B, L, _ = target_embedding.shape
        S, _ = source_embedding.shape
        H = self.n_heads

        target_embedding = self.query_projection(target_embedding).view(B, L, H, -1)
        source_embedding = self.key_projection(source_embedding).view(S, H, -1)
        value_embedding = self.value_projection(value_embedding).view(S, H, -1)

        out = self.reprogramming(target_embedding, source_embedding, value_embedding)

        out = out.reshape(B, L, -1)

        return self.out_projection(out)

    def reprogramming(self, target_embedding, source_embedding, value_embedding):
        B, L, H, E = target_embedding.shape
        
        scale = 1. / sqrt(E)

        scores = torch.einsum("blhe,she->bhls", target_embedding, source_embedding)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        reprogramming_embedding = torch.einsum("bhls,she->blhe", A, value_embedding)

        return reprogramming_embedding

class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        # x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x