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
from sklearn.cluster import KMeans
from utils.tools import sup_res_lines, fit_trendlines_single, fit_trendlines_high_low
from models.RevIN import RevIN

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
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.pos_embed = PositionalEmbedding(d_model=self.d_model)
        
        self.llama2 = LlamaModel.from_pretrained('meta-llama/Llama-2-13b-hf', token='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',output_attentions=True, output_hidden_states=True, load_in_4bit = True)
        # self.llama2 = AutoModel.from_pretrained("01-ai/Yi-6B", trust_remote_code=True)
        self.in_layer = nn.Linear(configs.seq_len, self.d_model)
        self.in_layer1 = nn.Linear(7, self.d_model)
        # self.in_layer2 = nn.Linear(4, self.d_model)

        self.token = LlamaTokenizer.from_pretrained('meta-llama/Llama-2-13b-hf', token='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')
        # self.token = AutoTokenizer.from_pretrained("01-ai/Yi-6B", trust_remote_code=True)

        self.revin_layer = RevIN(num_features=configs.num_features, affine = True, subtract_last = True)

        self.device = device
        self.in_layer_voc = nn.Linear(in_features= 32000,out_features= self.voc)
        self.attention = AttentionLayer(FullAttention(False, 5, attention_dropout=configs.dropout,output_attention= False,configs=configs), configs.d_model, configs.n_heads)
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

        # self.llama2.to(device, dtype = torch.bfloat16)
                
        # print(self.llama2.layers)
        # self.llama2.train()

        for layer in (self.in_layer,self.in_layer1, self.out_layer, self.in_layer_voc, self.attention, self.pos_embed, self.revin_layer):
            layer.float().to(device=device)
            layer.train()
        
        self.cnt = 0

    def forward(self, x, itr):
        close = x[:,:,-1]
        high = x[:,:,-3]
        low = x[:,:, -2]
        maxprice = torch.max(close[:,:])
        minprice = torch.min(close[:,:])
        meanval = torch.mean(close[0,:])
        SRlines = sup_res_lines(x[0,:,-1].cpu().detach().numpy())
        # [coefssup,coefsres], _ = fit_trendlines_single(x[0,-50:,0].cpu().detach().numpy())
        [coefssup,coefsres] = fit_trendlines_high_low(x[0,-50:,-3].cpu().detach().numpy(),x[0,-50:,-2].cpu().detach().numpy(),x[0,-50:,-1].cpu().detach().numpy())
        # x = torch.cumprod(x+1, dim = 1)
        B, L, M = x.shape
        # means = x.mean(1, keepdim=True).detach()
        # x = x - means
        # stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False)+ 1e-5).detach() 
        # x /= stdev
        
        x = self.revin_layer(x, 'norm')
       
        x = rearrange(x, 'b l m -> b m l')
        x1 = x[:,:,-7:]
        # x2 = x[:,:,-2:]
        out1 = self.in_layer1(x1)
        # out2 = self.in_layer2(x2)
        # print(self.in_layer(x))
        outputs = self.in_layer(x) 
        # print(outputs.shape)
        # print(lastvol)
        # print(lastRSI)
        # print(lastprice)

        prompt = f'This dataset is the Bitcoin daily price chart.'\
            # f'Below is the information about the input time series:' \
            # f'[Domain]: Bitcoin is the leading cryptocurrency. The bitcoin price is a highly volatile price chart which is globally on an upward trend although it oscillates between bull and bear market cycles that last around 1 to 2 years.' \
            # f'Each data point indicates the opening, high, low and closing price of Bitcoin as well as a moving average of 50 days. The volume, volatility, and relative strength index are also indicated.' \
            # f'The fear and greed index is also given, which provides a view on the overall sentiment surrounding bitcoin. Its value is  between 0 and 100 with lower values signifying fear and higher values greed. Pay attention to extreme values of this index which may indicate bitcoin is oversold or overbought.'\
            # f'The trend strength index is also given. A high value indicates a strong upward or downward trend. In this case the price is likely to keep following the trend. At low values of this index, the price is ranging, and trend reversals are more likely. During the periods of ranging, trend lines are useful to pinpoint trend reversals.'\
            # f'For each data point the trend lines from the previous 50 candles have been constructed and the normalized distance of the current price to the support and resistance line is given. When the price is close to a trend line, a trend reversal can be expected. After several tests on a trendline the price may also breakthrough.'\
            # f'Pay attention to price movement when the price is close to either trend line as this is where it is most predictable.'\
            # f'[Instructions]: Predict the detrended data for the next {self.pred_len} steps given the previous {self.seq_len} steps.'\
            # f'[Statistics]: The input has a minimum value of {minprice:.0f} and a maximum value of {maxprice:.0f}, with an average value of {meanval:.0f}.'\
            # f'The three most statistically prominent price levels are {float(SRlines[0]):.0f}, {float(SRlines[1]):.0f} and {float(SRlines[2]):.0f}.'
        
        prompt = self.token.encode(prompt, return_tensors = 'pt').to(self.device)
        prompt = self.embed(prompt)
        
        if self.pca == 0 :
            text_prot = self.in_layer_voc(self.voc_in.permute(1, 0)).permute(1,0)
        elif self.pca == 1 or self.pca == 2 :
            text_prot = self.voc_in
        text_prot = torch.broadcast_to(text_prot, (x.shape[0],-1,-1)).detach()
        outputs, attns = self.attention(outputs, text_prot, text_prot, None , None)
        # print(outputs.shape)
        prompt = torch.broadcast_to(prompt, (outputs.shape[0],-1,-1)).detach()
        outputs = torch.cat((prompt, outputs), dim = 1)
        outputs = outputs
        with torch.autocast(device_type= "cuda", dtype = torch.bfloat16):
            outputs = self.llama2(inputs_embeds=outputs).last_hidden_state
        outputs = self.out_layer(outputs)
        outputs = outputs[:,-12:,:]
        outputs = rearrange(outputs, 'b m l -> b l m')

        outputs = self.revin_layer(outputs, 'denorm')
        # outputs = outputs * stdev
        # outputs = outputs + means
        


        return outputs


