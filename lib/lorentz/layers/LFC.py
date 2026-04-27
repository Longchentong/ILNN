import torch
import torch.nn as nn

from lib.lorentz.manifold import CustomLorentz
from lib.GyroBN.Geometry.constantcurvature.utils.math_hyperboloid import gyroadd_radius,exp0
import share

class LorentzFullyConnected_old(nn.Module):
    """
        Modified Lorentz fully connected layer of Chen et al. (2022).

        Code modified from https://github.com/chenweize1998/fully-hyperbolic-nn

        args:
            manifold: Instance of Lorentz manifold
            in_features, out_features, bias: Same as nn.Linear
            init_scale: Scale parameter for internal normalization
            learn_scale: If scale parameter should be learnable
            normalize: If internal normalization should be applied
    """

    def __init__(
            self,
            manifold: CustomLorentz,
            in_features,
            out_features,
            bias=False,
            init_scale=None,
            learn_scale=False,
            normalize=False
        ):
        super(LorentzFullyConnected_old, self).__init__()
        self.manifold = manifold
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.normalize = normalize

        self.weight = nn.Linear(self.in_features, self.out_features, bias=bias)

        self.init_std = 0.02
        self.reset_parameters()

        # Scale for internal normalization
        if init_scale is not None:
            self.scale = nn.Parameter(torch.ones(()) * init_scale, requires_grad=learn_scale)
        else:
            self.scale = nn.Parameter(torch.ones(()) * 2.3, requires_grad=learn_scale)

    def forward(self, x):

        x = self.weight(x)
        x_space = x.narrow(-1, 1, x.shape[-1] - 1)

        if self.normalize:
            scale = x.narrow(-1, 0, 1).sigmoid() * self.scale.exp()
            square_norm = (x_space * x_space).sum(dim=-1, keepdim=True)

            mask = square_norm <= 1e-10

            square_norm[mask] = 1
            unit_length = x_space/torch.sqrt(square_norm)
            x_space = scale*unit_length

            x_time = torch.sqrt(scale**2 + self.manifold.k + 1e-5)
            x_time = x_time.masked_fill(mask, self.manifold.k.sqrt())

            mask = mask==False
            x_space = x_space * mask

            x = torch.cat([x_time, x_space], dim=-1)
        else:
            x = self.manifold.add_time(x_space)

        return x

    def reset_parameters(self):
        nn.init.uniform_(self.weight.weight, -self.init_std, self.init_std)

        if self.bias:
            nn.init.constant_(self.weight.bias, 0)




class PointToHyperplaneLorentzFC(nn.Module):
    """
    Distance-based Lorentz fully-connected layer based on the formulas provided.
    
    This implementation follows the mathematical formulas:
    - v_k(x) = (1/√(-K)) * sign(α) * β * |sinh^(-1)(√(-K) * α/β)|
    - α = cosh(√(-K)a) * ⟨z, x_s⟩ - sinh(√(-K)a) * ||z|| * x_t
    - β = √(||cosh(√(-K)a)z||² - (sinh(√(-K)a)||z||)²)
    - w_k = (1/√(-K)) * sinh(√(-K) * v_k(x))
    - y_0 = √(1/(-K) + ||w||²)
    - y = (y_0, w)
    
    Parameters:
    -----------
    manifold : CustomLorentz
        Lorentz manifold instance with curvature K < 0
    in_features : int
        Input feature dimension (n+1, including time coordinate)
    out_features : int
        Output feature dimension (m)
    eps : float
        Numerical stability constant
    """
    
    def __init__(self, manifold, in_features, out_features, eps=1e-9,    bias=False,
            init_scale=None,
            learn_scale=False,
            normalize=False):
        super().__init__()
        assert in_features >= 2, "input = [t, x1, …, xn]"
        
        self.manifold = manifold
        self.in_features = in_features
        self.out_features = out_features
        self.eps = eps
        self.dropout = 0.0
        self.normalize = normalize
        # print("self.normalize: ", self.normalize)
        self.z = nn.Parameter(torch.empty(out_features - 1, in_features - 1))  # (m-1, n)
        self.a = nn.Parameter(torch.zeros(out_features - 1))  # (m-1,)
        # self.normalize = normalize
        # if normalize:
            # self.bn = GyroBNH(shape=[channel[out_features -1],out_features - 1], model="Hyperboloid", K= self.manifold.k if self.manifold.k < 0 else -self.manifold.k, translate="Left")
        self.p = 0.0

        # print("share.share_b: ", share.share_b)
        if share.share_b:
            self.b = nn.Parameter(torch.zeros(out_features - 1))  # (m-1,)
        self.reset_parameters()
        
        if init_scale is not None:
            self.scale = nn.Parameter(torch.ones(()) * init_scale, requires_grad=learn_scale)
        else:
            self.scale = nn.Parameter(torch.ones(()) * 2.3, requires_grad=learn_scale)

    
    def reset_parameters(self):
        nn.init.normal_(self.z, mean=0.0, std=0.02)
        nn.init.zeros_(self.a)
        if share.share_b:
            nn.init.zeros_(self.b)
    
    def forward(self, x):
        """
        x : (..., n+1), point on the Lorentz manifold.
        returns y : (..., m+1), point on the Lorentz manifold.
        """

        if self.training and self.dropout > 1e-8:
            # print("dropout: ", self.dropout)
            mask = torch.bernoulli(torch.full_like(self.z, 1 - self.dropout))
            mask_z = self.z * mask
            mask = torch.bernoulli(torch.full_like(self.a, 1 - self.dropout))
            mask_a = self.a * mask
        else:
            mask_z = self.z
            mask_a = self.a

        K = self.manifold.k  # K<0
        if K > 0:
            K = -K  
        # print("mask_z: ", mask_z.shape)
        # print("mask_a: ", mask_a.shape)
        # print("x: ", x.shape)
        sqrt_neg_K = torch.sqrt(-K)  # √(-K)
        inv_sqrt_neg_K = 1.0 / sqrt_neg_K  # 1/√(-K)
        
        x_t = x[..., 0]  # (...,) time
        x_s = x[..., 1:]  # (..., n) space
        
        # Compute ||z_k||.
        norm_z = torch.linalg.norm(mask_z, dim=-1)  # (m,)
        
        # Compute cosh(sqrt(-K) a) and sinh(sqrt(-K) a).
        cosh_term = torch.cosh(sqrt_neg_K * mask_a)  # (m,)
        sinh_term = torch.sinh(sqrt_neg_K * mask_a)  # (m,)
        
        # alpha = cosh(sqrt(-K) a)<z, x_s> - sinh(sqrt(-K) a)||z||x_t.
        z_dot_xs = torch.einsum('mk,...k->...m', mask_z, x_s)  # (..., m)
        
        alpha = cosh_term * z_dot_xs - sinh_term * norm_z * x_t.unsqueeze(-1)  # (..., m)
        
        # beta = sqrt(||cosh(sqrt(-K) a)z||^2 - (sinh(sqrt(-K) a)||z||)^2).
        term1 = (cosh_term * norm_z) ** 2  # (m,)
        term2 = (sinh_term * norm_z) ** 2  # (m,)
        beta = torch.sqrt(term1 - term2 + self.eps)  # (m,)
        
        # Compute sqrt(-K) alpha / beta.
        ratio = sqrt_neg_K * alpha / beta  # (..., m)
        
        # Compute asinh(sqrt(-K) alpha / beta).
        asinh_term = torch.asinh(ratio)  # (..., m)
        
        # v_k(x) = sign(alpha) beta |asinh(sqrt(-K) alpha / beta)| / sqrt(-K).
        v = inv_sqrt_neg_K * torch.sign(alpha) * beta * torch.abs(asinh_term)  # (..., m)
        # v = torch.tanh(v)
        v = v.clamp(min=-10, max=10)
        # print(v)
        # Spatial coordinates w_k = sinh(sqrt(-K) v_k(x)) / sqrt(-K).
        w = inv_sqrt_neg_K * torch.sinh(sqrt_neg_K * v)  # (..., m)
        
        # Time coordinate y_0 = sqrt(1 / (-K) + ||w||^2).
        w_squared_norm = (w ** 2).sum(dim=-1, keepdim=True)  # (..., 1)
        y_0 = torch.sqrt(1.0 / (-K) + w_squared_norm + self.eps)  # (..., 1)
        
        y = torch.cat([y_0, w], dim=-1)  # (..., m+1)

        if share.share_b:
            L_b = torch.zeros_like(y)
            L_b[..., 1:] = self.b
            y = gyroadd_radius(y, exp0(L_b, K), K)

        if self.normalize:
            x = y
            x_space = x.narrow(-1, 1, x.shape[-1] - 1)

            scale = x.narrow(-1, 0, 1).sigmoid() * self.scale.exp()
            square_norm = (x_space * x_space).sum(dim=-1, keepdim=True)

            mask = square_norm <= 1e-10

            square_norm[mask] = 1
            unit_length = x_space/torch.sqrt(square_norm)
            x_space = scale*unit_length

            x_time = torch.sqrt(scale**2 + self.manifold.k + 1e-5)
            x_time = x_time.masked_fill(mask, self.manifold.k.sqrt())

            mask = mask==False
            x_space = x_space * mask

            x = torch.cat([x_time, x_space], dim=-1)
            return x  

        return y

LorentzFullyConnected_new = PointToHyperplaneLorentzFC
LorentzFullyConnected = PointToHyperplaneLorentzFC if share.share_FC_new else LorentzFullyConnected_old
