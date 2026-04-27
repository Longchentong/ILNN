import torch
import torch.nn as nn
import torch.nn.functional as F

import math

from lib.lorentz.manifold import CustomLorentz
from lib.lorentz.layers import LorentzFullyConnected


def lorentz_log_radius_concat(
        patches: torch.Tensor,
        in_channels: int,
        kernel_len: int,
        k: torch.Tensor,
        eps: float = 1e-9,
) -> torch.Tensor:
    """Concatenate Lorentz patch blocks with log-radius alignment.

    ``patches`` is channel-major, as returned by ``unfold``:
    ``[time_1..time_N, space_channel_1_block, ...]``. The output is a single
    Lorentz vector with one time coordinate and all spatial blocks stacked.
    """
    batch, windows, width = patches.shape
    block_dim = in_channels - 1
    expected_width = kernel_len + kernel_len * block_dim
    if width != expected_width:
        raise ValueError(f"expected patch width {expected_width}, got {width}")

    k = k.to(device=patches.device, dtype=patches.dtype)
    sqrt_k = torch.sqrt(k.clamp_min(eps))

    patch_time = torch.clamp(patches[..., :kernel_len], min=sqrt_k)
    patch_space = patches[..., kernel_len:]

    n_total = kernel_len * block_dim
    if block_dim > 0 and n_total != block_dim:
        digamma = getattr(torch.special, "digamma", torch.digamma)
        scale = torch.exp(
            0.5 * (
                digamma(torch.as_tensor(n_total / 2.0, device=patches.device, dtype=patches.dtype))
                - digamma(torch.as_tensor(block_dim / 2.0, device=patches.device, dtype=patches.dtype))
            )
        )
    else:
        scale = patches.new_tensor(1.0)

    patch_space = patch_space.reshape(batch, windows, block_dim, kernel_len).transpose(-1, -2)
    patch_space = patch_space * scale
    patch_space = patch_space.transpose(-1, -2).reshape(batch, windows, n_total)

    summed_space_norm_sq = (patch_time.pow(2) - k).sum(dim=-1, keepdim=True)
    time = torch.sqrt(torch.clamp_min(k + scale.pow(2) * summed_space_norm_sq, k + eps))
    return torch.cat([time, patch_space], dim=-1)

class LorentzConv1d(nn.Module):
    """ Implements a fully hyperbolic 1D convolutional layer using the Lorentz model.

    Args:
        manifold: Instance of Lorentz manifold
        in_channels, out_channels, kernel_size, stride, padding, bias: Same as nn.Conv1d
        LFC_normalize: If Chen et al.'s internal normalization should be used in LFC 
    """
    def __init__(
            self,
            manifold: CustomLorentz,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            bias=True,
            LFC_normalize=False
    ):
        super(LorentzConv1d, self).__init__()

        self.manifold = manifold
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        lin_features = (self.in_channels - 1) * self.kernel_size + 1

        self.linearized_kernel = LorentzFullyConnected(
            manifold,
            lin_features, 
            self.out_channels, 
            bias=bias,
            normalize=LFC_normalize
        )

    def forward(self, x):
        """ x has to be in channel-last representation -> Shape = bs x len x C """
        bsz = x.shape[0]

        # origin padding
        x = F.pad(x, (0, 0, self.padding, self.padding))
        x[..., 0].clamp_(min=self.manifold.k.sqrt()) 

        patches = x.unfold(1, self.kernel_size, self.stride).contiguous()
        patches = patches.reshape(bsz, patches.shape[1], -1)
        patches_pre_kernel = lorentz_log_radius_concat(
            patches=patches,
            in_channels=self.in_channels,
            kernel_len=self.kernel_size,
            k=self.manifold.k,
        )

        out = self.linearized_kernel(patches_pre_kernel)

        return out


class LorentzConv2d(nn.Module):
    """ Implements a fully hyperbolic 2D convolutional layer using the Lorentz model.

    Args:
        manifold: Instance of Lorentz manifold
        in_channels, out_channels, kernel_size, stride, padding, dilation, bias: Same as nn.Conv2d (dilation not tested)
        LFC_normalize: If Chen et al.'s internal normalization should be used in LFC 
    """
    def __init__(
            self,
            manifold: CustomLorentz,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            bias=True,
            LFC_normalize=False
    ):
        super(LorentzConv2d, self).__init__()

        self.manifold = manifold
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.bias = bias

        if isinstance(stride, int):
            self.stride = (stride, stride)
        else:
            self.stride = stride

        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size

        if isinstance(padding, int):
            self.padding = (padding, padding)
        else:
            self.padding = padding

        if isinstance(dilation, int):
            self.dilation = (dilation, dilation)
        else:
            self.dilation = dilation

        self.kernel_len = self.kernel_size[0] * self.kernel_size[1]

        lin_features = ((self.in_channels - 1) * self.kernel_size[0] * self.kernel_size[1]) + 1

        self.linearized_kernel = LorentzFullyConnected(
            manifold,
            lin_features, 
            self.out_channels, 
            bias=bias,
            normalize=LFC_normalize
        )
        self.unfold = torch.nn.Unfold(kernel_size=(self.kernel_size[0], self.kernel_size[1]), dilation=dilation, padding=padding, stride=stride)

        self.reset_parameters()

    def reset_parameters(self):
        if hasattr(self.linearized_kernel, "reset_parameters"):
            self.linearized_kernel.reset_parameters()
            return

        stdv = math.sqrt(2.0 / ((self.in_channels - 1) * self.kernel_size[0] * self.kernel_size[1]))
        if hasattr(self.linearized_kernel, "weight") and hasattr(self.linearized_kernel.weight, "weight"):
            self.linearized_kernel.weight.weight.data.uniform_(-stdv, stdv)
            if self.bias and self.linearized_kernel.weight.bias is not None:
                self.linearized_kernel.weight.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        """ x has to be in channel-last representation -> Shape = bs x H x W x C """
        bsz = x.shape[0]
        h, w = x.shape[1:3]

        h_out = math.floor(
            (h + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) / self.stride[0] + 1)
        w_out = math.floor(
            (w + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1) / self.stride[1] + 1)

        x = x.permute(0, 3, 1, 2)

        patches = self.unfold(x)  # batch_size, channels * elements/window, windows
        patches = patches.permute(0, 2, 1)

        patches_pre_kernel = lorentz_log_radius_concat(
            patches=patches,
            in_channels=self.in_channels,
            kernel_len=self.kernel_len,
            k=self.manifold.k,
        )

        out = self.linearized_kernel(patches_pre_kernel)
        out = out.view(bsz, h_out, w_out, self.out_channels)

        return out

class LorentzConvTranspose2d(nn.Module):
    """ Implements a fully hyperbolic 2D transposed convolutional layer using the Lorentz model.

    Args:
        manifold: Instance of Lorentz manifold
        in_channels, out_channels, kernel_size, stride, padding, output_padding, bias: Same as nn.ConvTranspose2d
        LFC_normalize: If Chen et al.'s internal normalization should be used in LFC 
    """
    def __init__(
            self, 
            manifold: CustomLorentz, 
            in_channels, 
            out_channels, 
            kernel_size, 
            stride=1, 
            padding=0, 
            output_padding=0, 
            bias=True,
            LFC_normalize=False
        ):
        super(LorentzConvTranspose2d, self).__init__()

        self.manifold = manifold
        self.in_channels = in_channels
        self.out_channels = out_channels

        if isinstance(stride, int):
            self.stride = (stride, stride)
        else:
            self.stride = stride

        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size

        if isinstance(padding, int):
            self.padding = (padding, padding)
        else:
            self.padding = padding

        if isinstance(output_padding, int):
            self.output_padding = (output_padding, output_padding)
        else:
            self.output_padding = output_padding

        padding_implicit = [0,0]
        padding_implicit[0] = kernel_size - self.padding[0] - 1 # Ensure padding > kernel_size
        padding_implicit[1] = kernel_size - self.padding[1] - 1 # Ensure padding > kernel_size

        self.pad_weight = nn.Parameter(F.pad(torch.ones((self.in_channels,1,1,1)),(1,1,1,1)), requires_grad=False)

        self.conv = LorentzConv2d(
            manifold=manifold, 
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            stride=1, 
            padding=padding_implicit, 
            bias=bias, 
            LFC_normalize=LFC_normalize
        )

    def forward(self, x):
        """ x has to be in channel last representation -> Shape = bs x H x W x C """
        if self.stride[0] > 1 or self.stride[1] > 1:
            # Insert hyperbolic origin vectors between features
            x = x.permute(0,3,1,2)
            # -> Insert zero vectors
            x = F.conv_transpose2d(x, self.pad_weight,stride=self.stride,padding=1, groups=self.in_channels)
            x = x.permute(0,2,3,1)
            x[..., 0].clamp_(min=self.manifold.k.sqrt())

        x = self.conv(x)

        if self.output_padding[0] > 0 or self.output_padding[1] > 0:
            x = F.pad(x, pad=(0, self.output_padding[1], 0, self.output_padding[0])) # Pad one side of each dimension (bottom+right) (see PyTorch documentation)
            x[..., 0].clamp_(min=self.manifold.k.sqrt()) # Fix origin padding

        return x


class LorentzDropout(nn.Module):
    """Mask Lorentz coordinates and project back to the hyperboloid."""
    def __init__(self, manifold, p: float = 0.5):
        super().__init__()
        self.manifold = manifold
        self.p = p
        self.dropout = nn.Dropout(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # print("dropout: ", self.p)
        if not self.training or self.p < 1e-8:
            return x
        v = self.dropout(x)
        x_drop = self.manifold.projx(v)
        return x_drop
