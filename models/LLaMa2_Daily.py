import numpy as np
import torch
import torch.nn as nn
from torch import optim
from transformers import LlamaConfig, LlamaModel, LlamaTokenizer
from einops import rearrange
from models.PatchTST import AttentionLayer, FullAttention
from embed import PositionalEmbedding
import sys
from utils.tools import sup_res_lines

class LlamaDaily(nn.Module):
    def __init__(self, configs, device):
        super(LlamaDaily, self).__init__()
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
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.pos_embed = PositionalEmbedding(d_model=self.d_model)
        if configs.isllama :
            if configs.pretrain :
                self.llama2 = LlamaModel.from_pretrained('meta-llama/Llama-2-13b-hf', token='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',output_attentions=True, output_hidden_states=True)
        self.in_layer = nn.Linear(configs.seq_len, self.d_model)
        self.in_layer1 = nn.Linear(7, self.d_model)
        self.in_layer2 = nn.Linear(2, self.d_model)

        self.token = LlamaTokenizer.from_pretrained('meta-llama/Llama-2-13b-hf', token='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')
        self.device = device
        self.in_layer_voc = nn.Linear(in_features= 32000,out_features= self.voc)
        self.attention = AttentionLayer(FullAttention(False, 5, attention_dropout=configs.dropout,output_attention= False,configs=configs), configs.d_model, configs.n_heads)
        self.out_layer = nn.Linear(configs.d_model, configs.pred_len, bias = True)
        # self.llama2 = self.llama2[:configs.gpt_layers]
        self.embed = self.llama2.get_input_embeddings()
        self.voc_in = self.embed(torch.tensor(range(32000)))
        self.voc_in=self.voc_in.to(device = device)

        if configs.freeze and configs.pretrain:
            for i, (name, param) in enumerate(self.llama2.named_parameters()):
                if 'layernorm' in name or 'norm' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        
        self.llama2.layers = self.llama2.layers[:configs.gpt_layers]
        self.llama2.to(device, dtype = torch.bfloat16)
        # self.llama2.train()

        for layer in (self.in_layer,self.in_layer1,self.in_layer2, self.out_layer, self.in_layer_voc, self.attention, self.pos_embed):
            layer.float().to(device=device)
            layer.train()
        
        self.cnt = 0

    def forward(self, x, itr):
        info = x[:,:,:3]
        maxprice = torch.max(info[:,:,0])
        minprice = torch.min(info[:,:,0])
        SRlines = sup_res_lines(x[0,:,0].cpu().detach().numpy())
        x = x[:,:,3:]
        # x = torch.cumprod(x+1, dim = 1)
        B, L, M = x.shape
        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False)+ 1e-5).detach() 
        x /= stdev
        
       
        x = rearrange(x, 'b l m -> b m l')
        x1 = x[:,:,-7:]
        x2 = x[:,:,-2:]
        out1 = self.in_layer1(x1)
        out2 = self.in_layer2(x2)
        # print(self.in_layer(x))
        outputs = self.in_layer(x) + out1 + out2
        # print(outputs.shape)
        # print(lastvol)
        # print(lastRSI)
        # print(lastprice)

        prompt = f'This dataset is the Bitcoin daily price chart.'\
            f' Each data point indicates the daily percent change in the opening, high, low and closing price of Bitcoin as well as an exponential moving average of 50 days.' \
            f'Below is the information about the input time series:' \
            f'[Domain]: Bitcoin is the leading cryptocurrency, the bitcoin price is a highly volatile price chart which is globally on an upward trend.' \
            f'[Instructions]: Predict the percent change in price for the next {self.pred_len} steps given the previous {self.seq_len} steps.'\
            f'[Statistics]: The input has a minimum closing price of {minprice:.0f} dollars and a maximum closing price of {maxprice:.0f} dollars.' \
            f'The three most important support and resistance prices are {float(SRlines[0]):.0f}, {float(SRlines[1]):.0f} and {float(SRlines[2]):.0f} dollars.'
        prompt = self.token.encode(prompt, return_tensors = 'pt').to(self.device)
        prompt = self.embed(prompt)
        text_prot = self.in_layer_voc(self.voc_in.permute(1, 0)).permute(1,0)
        text_prot = torch.broadcast_to(text_prot, (x.shape[0],-1,-1)).detach()
        outputs, attns = self.attention(outputs, text_prot, text_prot, None , None)
        prompt = torch.broadcast_to(prompt, (outputs.shape[0],-1,-1)).detach()
        outputs = torch.cat((prompt, outputs), dim = 1)

        if self.isllama:
            with torch.autocast(device_type= "cuda", dtype = torch.bfloat16):
                outputs = self.llama2(inputs_embeds=outputs).last_hidden_state
        outputs = self.out_layer(outputs)
        outputs = outputs[:,-5:, :]
        outputs = rearrange(outputs, 'b m l -> b l m')
        outputs = outputs * stdev
        outputs = outputs + means
        return outputs


