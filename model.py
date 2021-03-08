import torch 
import torch.nn as nn 
from TorchSUL import Model as M 

class Embedder(M.Model):
    def initialize(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()
    
    def create_embedding_fn(self):
        d = self.kwargs['input_dims']
        outdim = 0

        if self.kwargs['include_input']:
            outdim += d 

        max_freq = self.kwargs['max_freq_log2']
        n_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, n_freqs)
        else:
            freq_bands = torch.linspace(2.**0, 2.**max_freq, n_freqs)
        
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                outdim += d 

        self.register_buffer('freq_bands', freq_bands)
        self.outdim = outdim

    def forward(self, x):
        outs = []
        if self.kwargs['include_input']:
            outs.append(x)
        for freq in self.freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                outs.append(p_fn(x * freq))
        result = torch.cat(outs, dim=1)
        return result

def get_embedder(multires):
    embed_kwargs = {
        'include_input': True,
        'input_dims': 3,
        'max_freq_log2': multires-1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder = Embedder(**embed_kwargs)
    return embedder, embedder.outdim

class NERF(M.Model):
    def initialize(self, D=8, W=256, skips=[4]):
        # use_viewdirs = True 
        self.layers = nn.ModuleList()
        for i in range(D):
            self.layers.append(M.Dense(W, activation=M.PARAM_RELU))
        
        self.skips = skips

        self.alpha_fc = M.Dense(1)
        self.bottleneck = M.Dense(256)
        self.hidden = M.Dense(W//2)
        self.out_fc = M.Dense(3)

    def forward(self, embed, embeddir):
        x = embed 
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            if i in self.skips:
                x = torch.cat([embed, x], dim=1)
        
        alpha_out = self.alpha_fc(x)

        bottleneck = self.bottleneck(x)
        x = torch.cat([bottleneck, embeddir],dim=1)
        out = self.out_fc(x)
        out = torch.cat([out, alpha_out], -1)
        return out

def get_nerf_model(D, W, skips, embed_chn, embeddir_chn):
    dumb_embed = torch.zeros(1, embed_chn)
    dumb_embeddir = torch.zeros(1, embeddir_chn)
    nerf = NERF(D,W,skips)
    nerf(dumb_embed, dumb_embeddir)
    return nerf
