import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.geoopt import ManifoldParameter
from lib.lorentz.manifold import CustomLorentz
from lib.GyroBN.GyroBNH import GyroBNH
import share


class LorentzBatchNorm(nn.Module):
    """ Lorentz Batch Normalization with Centroid and Fréchet variance
    """
    def __init__(self, manifold: CustomLorentz, num_features: int):
        super(LorentzBatchNorm, self).__init__()
        self.manifold = manifold

        self.beta = ManifoldParameter(self.manifold.origin(num_features), manifold=self.manifold)
        self.gamma = torch.nn.Parameter(torch.ones((1,)))

        self.eps = 1e-5

        # running statistics
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones((1,)))

    def forward(self, x, momentum=0.1):
        assert (len(x.shape)==2) or (len(x.shape)==3), "Wrong input shape in Lorentz batch normalization."

        beta = self.beta

        if self.training:
            # Compute batch mean
            mean = self.manifold.centroid(x)
            if len(x.shape) == 3:
                mean = self.manifold.centroid(mean)

            # Transport batch to origin (center batch)
            x_T = self.manifold.logmap(mean, x)
            x_T = self.manifold.transp0back(mean, x_T)

            # Compute Fréchet variance
            if len(x.shape) == 3:
                var = torch.mean(torch.norm(x_T, dim=-1), dim=(0,1))
            else:
                var = torch.mean(torch.norm(x_T, dim=-1), dim=0)

            # Rescale batch
            x_T = x_T*(self.gamma/(var+self.eps))

            # Transport batch to learned mean
            x_T = self.manifold.transp0(beta, x_T)
            output = self.manifold.expmap(beta, x_T)

            # Save running parameters
            with torch.no_grad():
                running_mean = self.manifold.expmap0(self.running_mean)
                means = torch.concat((running_mean.unsqueeze(0), mean.detach().unsqueeze(0)), dim=0)
                self.running_mean.copy_(self.manifold.logmap0(self.manifold.centroid(means, w=torch.tensor(((1-momentum), momentum), device=means.device))))
                self.running_var.copy_((1 - momentum)*self.running_var + momentum*var.detach())

        else:
            # Transport batch to origin (center batch)
            running_mean = self.manifold.expmap0(self.running_mean)
            x_T = self.manifold.logmap(running_mean, x)
            x_T = self.manifold.transp0back(running_mean, x_T)

            # Rescale batch
            x_T = x_T*(self.gamma/(self.running_var+self.eps))

            # Transport batch to learned mean
            x_T = self.manifold.transp0(beta, x_T)
            output = self.manifold.expmap(beta, x_T)

        return output

class LorentzBatchNorm1d_old(LorentzBatchNorm):
    """ 1D Lorentz Batch Normalization with Centroid and Fréchet variance
    """
    def __init__(self, manifold: CustomLorentz, num_features: int):
        super(LorentzBatchNorm1d_old, self).__init__(manifold, num_features)

    def forward(self, x, momentum=0.1):
        return super(LorentzBatchNorm1d_old, self).forward(x, momentum)

class LorentzBatchNorm2d_old(LorentzBatchNorm):
    """ 2D Lorentz Batch Normalization with Centroid and Fréchet variance
    """
    def __init__(self, manifold: CustomLorentz, num_channels: int):
        super(LorentzBatchNorm2d_old, self).__init__(manifold, num_channels)

    def forward(self, x, momentum=0.1):
        """ x has to be in channel last representation -> Shape = bs x H x W x C """
        bs, h, w, c = x.shape
        x = x.view(bs, -1, c)
        x = super(LorentzBatchNorm2d_old, self).forward(x, momentum)
        x = x.reshape(bs, h, w, c)

        return x


class LorentzBatchNorm1d_new(GyroBNH):
    def __init__(self, manifold: CustomLorentz, num_features: int):
        # self.num_features = num_features
        super(LorentzBatchNorm1d_new, self).__init__(shape=[num_features-1], 
                                                 model="Hyperboloid", 
                                                 K= manifold.k if manifold.k < 0 else -manifold.k, 
                                                 translate="Left",
                                                 max_iter=4
                                                 )
    def forward(self, x):
        x_orign = x.dtype
        if x.dtype != self.weight.dtype:
            x = x.to(self.weight.dtype)
        
        if len(x.shape) == 3:
            orign = x.shape
            x = x.view(-1, x.shape[-1])
            x = super(LorentzBatchNorm1d_new, self).forward(x)
            x = x.view(orign)
        else:
            x = super(LorentzBatchNorm1d_new, self).forward(x)
        x = x.to(x_orign)
        return x

class LorentzBatchNorm2d_new(GyroBNH):
    def __init__(self, manifold: CustomLorentz, num_channels: int):
        # self.num_channels = num_channels
        # print("manifold.k: ", manifold.k)
        super(LorentzBatchNorm2d_new, self).__init__(shape=[num_channels-1], model="Hyperboloid", 
                                                 K= manifold.k if manifold.k < 0 else -manifold.k, translate="Left",
                                                 max_iter=4
                                                 )
    def forward(self, x):   
        x_orign = x.dtype
        if x.dtype != self.weight.dtype:
            x = x.to(self.weight.dtype)

        bs, h, w, c = x.shape
        x = x.view(-1, c) 
        # print("GBN")
        # if torch.isnan(x).any():
            # print(f"Input shape: {x.shape}")
            # print(f"Input range: [{x.min():.6f}, {x.max():.6f}]")
            # print(f"Input has NaN: {torch.isnan(x).any()}")
            
        x = super(LorentzBatchNorm2d_new, self).forward(x)
        # if torch.isnan(x).any():
            # print(f"Output shape: {x.shape}")
            # print(f"Output range: [{x.min():.6f}, {x.max():.6f}]")
            # print(f"Output has NaN: {torch.isnan(x).any()}")
    
        x = x.view(bs, h, w, c)
        x = x.to(x_orign)
        return x
    

LorentzBatchNorm2d = LorentzBatchNorm2d_new if share.share_bn_new else LorentzBatchNorm2d_old
LorentzBatchNorm1d = LorentzBatchNorm1d_new if share.share_bn_new else LorentzBatchNorm1d_old
