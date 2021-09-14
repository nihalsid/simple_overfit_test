import torch
import torch.nn as nn
import torch.nn.functional as F


class NetTexture(nn.Module):

    def __init__(self, sample_point_dim, texture_features, n_layers=8, n_freq=10, ngf=256):
        super(NetTexture, self).__init__()
        self.ngf = ngf
        self.layers = n_layers
        self.skips = [4]

        multires = n_freq
        self.embedder, self.embedder_out_dim = get_embedder(multires, input_dims=sample_point_dim, i=0)

        print('sample_point_dim: ', sample_point_dim)
        print('embedder_out_dim: ', self.embedder_out_dim)

        self.pts_linears = nn.ModuleList(
            [nn.Conv2d(self.embedder_out_dim, self.ngf, kernel_size=1, stride=1, padding=0)] +
            [nn.Conv2d(self.ngf, self.ngf, kernel_size=1, stride=1, padding=0) if i not in self.skips else nn.Conv2d(self.ngf + self.embedder_out_dim , self.ngf, kernel_size=1, stride=1, padding=0) for i in range(self.layers-1)])
        self.output_linear = nn.Conv2d(self.ngf, texture_features, kernel_size=1, stride=1, padding=0)
        self.simple_lin = nn.Conv2d(sample_point_dim, texture_features, kernel_size=1, stride=1, padding=0)

    def forward(self, input_pts):
        simple_net = False
        if simple_net:
            return self.simple_lin(input_pts)
        else:
            # input_pts in [-1,1]
            input_embed = self.embedder(input_pts)
            h = input_embed
            for i, l in enumerate(self.pts_linears):
                h = self.pts_linears[i](h)
                h = F.leaky_relu(h)
                if i in self.skips:
                    h = torch.cat([input_embed, h], 1)
            return self.output_linear(h)


class Embedder:
    def __init__(self, input_dims=3, include_input=True, max_freq_log2=10 - 1, num_freqs=10, log_sampling=True, periodic_fns=[torch.sin, torch.cos]):
        self.input_dims = input_dims
        self.include_input = include_input
        self.max_freq_log2 = max_freq_log2
        self.num_freqs = num_freqs
        self.log_sampling = log_sampling
        self.periodic_fns = periodic_fns

        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.input_dims
        out_dim = 0
        if self.include_input:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.max_freq_log2
        N_freqs = self.num_freqs

        if self.log_sampling:
            freq_bands = 2. ** torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.periodic_fns:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], 1)


PI = 3.141592


def get_embedder(multires, input_dims=3, i=0):
    if i == -1:
        return nn.Identity(), input_dims

    embed_kwargs = {
        'input_dims': input_dims,
        'include_input': True,
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj: eo.embed(PI * x)
    return embed, embedder_obj.out_dim