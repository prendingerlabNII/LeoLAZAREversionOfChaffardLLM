import numpy as np
import torch
import torch.nn as nn
from torch import optim
from transformers import LlamaConfig, LlamaModel, LlamaTokenizer, AutoModel, AutoTokenizer
from einops import rearrange
from models.PatchTST import AttentionLayer, FullAttention
from embed import PositionalEmbedding
import sys
from sklearn.decomposition import PCA
from airllm import AirLLMLlama2
from utils.tools import sup_res_lines, fit_trendlines_single, fit_trendlines_high_low
from models.RevIN import RevIN

class Llama(nn.Module):
    def __init__(self, configs, device):
        # TOKENIZER_PARRALLELISM = True
        super(Llama, self).__init__()
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
        pca = PCA()

        
        self.llama2 = LlamaModel.from_pretrained('meta-llama/Llama-2-13b-hf', token='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',output_attentions=True, output_hidden_states=True)
        # self.llama2 = AutoModel.from_pretrained("01-ai/Yi-6B", trust_remote_code=True)
        self.in_layer = nn.Linear(configs.patch_size, self.d_model)
        self.in_layer1 = nn.Linear(7, self.d_model)
        # self.in_layer2 = nn.Linear(4, self.d_model)

        self.token = LlamaTokenizer.from_pretrained('meta-llama/Llama-2-13b-hf', token='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')
        # self.token = AutoTokenizer.from_pretrained("01-ai/Yi-6B", trust_remote_code=True)

        self.device = device
        self.in_layer_voc = nn.Linear(in_features= 32000,out_features= self.voc)
        self.attention = AttentionLayer(FullAttention(False, 5, attention_dropout=configs.dropout,output_attention= False,configs=configs), configs.d_model, configs.n_heads)
        self.out_layer = nn.Linear(configs.d_model * self.patch_num, configs.pred_len)
        self.embed = self.llama2.get_input_embeddings()
        self.voc_in = self.embed(torch.tensor(range(32000)))
        self.pca = configs.pca
        if self.pca == 1 :
            self.voc_in = self.voc_in.detach().numpy()
            self.voc_in = torch.tensor(pca.fit_transform(X = self.voc_in)[:configs.voc])

        self.voc_in=self.voc_in.to(device = device)
        self.revin_layer = RevIN(num_features = configs.num_features, affine = True)
    
        for i, (name, param) in enumerate(self.llama2.named_parameters()):
            if 'layernorm' in name or 'norm' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        
        self.llama2.layers = self.llama2.layers[:configs.gpt_layers]
        self.llama2.float().to(device)
        # self.llama2.train()

        for layer in (self.in_layer,self.in_layer1, self.out_layer, self.in_layer_voc, self.attention, self.pos_embed, self.revin_layer):
            layer.float().to(device=device)
            layer.train()
        
        self.cnt = 0

    def forward(self, x, itr):

        B, L, M = x.shape
        x = self.revin_layer(x, 'norm')
        
       
        x = rearrange(x, 'b l m -> b m l')
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_size, step=self.stride)
        x = rearrange(x, 'b m n p -> (b m) n p')
        outputs = self.in_layer(x)
        min_val = torch.min(x[:,-1,:]).detach()
        max_val = torch.max(x[:,-1,:]).detach()
        mean_val = torch.mean(x[:,-1,:]).detach()
            
        prompt = f'This dataset is the Bitcoin daily price chart.'\
            f'Below is the information about the input time series:' \
            f'[Domain]: Bitcoin is the leading cryptocurrency. The bitcoin price is a highly volatile price chart which is globally on an upward trend although it oscillates between bull and bear market cycles that last around 1 to 2 years.' \
            f'Each data point indicates the opening, high, low and closing price of Bitcoin as well as a moving average of 50 days. The volume, volatility, and relative strength index are also indicated.' \
            f'The fear and greed index is also given, which provides a view on the overall sentiment surrounding bitcoin. Its value is  between 0 and 100 with lower values signifying fear and higher values greed. Pay attention to extreme values of this index which may indicate bitcoin is oversold or overbought.'\
            f'The trend strength index is also given. A high value indicates a strong upward or downward trend. In this case the price is likely to keep following the trend. At low values of this index, the price is ranging, and trend reversals are more likely. During the periods of ranging, trend lines are useful to pinpoint trend reversals.'\
            f'For each data point the trend lines from the previous 50 candles have been constructed and the normalized distance of the current price to the support and resistance line is given. When the price is close to a trend line, a trend reversal can be expected. After several tests on a trendline the price may also breakthrough.'\
            f'Pay attention to price movement when the price is close to either trend line as this is where it is most predictable.'\
            f'[Instructions]: Predict the detrended data for the next {self.pred_len} steps given the previous {self.seq_len} steps.'\
            f'[Statistics]: The input has a minimum value of {min_val:.0f} and a maximum value of {max_val:.0f}, with an average value of {mean_val:.0f}.'
            # f'The three most statistically prominent price levels are {float(SRlines[0]):.0f}, {float(SRlines[1]):.0f} and {float(SRlines[2]):.0f}.'
    
        prompt = self.token.encode(prompt, return_tensors = 'pt').to(self.device)
        prompt = self.embed(prompt)
        
        if self.pca == 0 :
            text_prot = self.in_layer_voc(self.voc_in.permute(1, 0)).permute(1,0)
        elif self.pca == 1 :
            text_prot = self.voc_in
        text_prot = torch.broadcast_to(text_prot, (x.shape[0],-1,-1)).detach()
        outputs, attns = self.attention(outputs, text_prot, text_prot, None , None)
        prompt = torch.broadcast_to(prompt, (outputs.shape[0],-1,-1)).detach()
        # outputs = torch.cat((prompt, outputs), dim = 1)
        outputs = outputs
        with torch.autocast(device_type= "cuda", dtype = torch.bfloat16):
            outputs = self.llama2(inputs_embeds=outputs).last_hidden_state
        outputs = outputs[:,-self.patch_num:,:]
        outputs = self.out_layer(outputs.reshape(B*M, -1))
        outputs = rearrange(outputs, '(b m) l -> b l m', b=B)
        outputs = self.revin_layer(outputs, 'denorm')
        
        return outputs


