import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import share
from lib.lorentz.manifold import CustomLorentz
from lib.lorentz.layers import (
    LorentzBatchNorm1d,
    LorentzConv1d,
    LorentzConv2d,
    LorentzDropout,
    LorentzFullyConnected,
)


def main():
    torch.manual_seed(0)
    manifold = CustomLorentz(k=1.0)
    share.share_manifold = manifold

    x = manifold.projx(torch.randn(4, 6))
    fc = LorentzFullyConnected(manifold, in_features=6, out_features=5)
    y = fc(x)
    assert y.shape == (4, 5)
    assert torch.isfinite(y).all()

    bn = LorentzBatchNorm1d(manifold, num_features=5)
    z = bn(y)
    assert z.shape == (4, 5)
    assert torch.isfinite(z).all()

    seq = manifold.projx(torch.randn(2, 8, 6))
    conv = LorentzConv1d(manifold, in_channels=6, out_channels=7, kernel_size=3, padding=1)
    out = conv(seq)
    assert out.shape == (2, 8, 7)
    assert torch.isfinite(out).all()

    img = manifold.projx(torch.randn(2, 8, 8, 4))
    conv2d = LorentzConv2d(manifold, in_channels=4, out_channels=6, kernel_size=3, padding=1)
    img_out = conv2d(img)
    assert img_out.shape == (2, 8, 8, 6)
    assert torch.isfinite(img_out).all()

    drop = LorentzDropout(manifold, p=0.1)
    drop.train()
    dropped = drop(out)
    assert dropped.shape == out.shape
    assert torch.isfinite(dropped).all()

    print("ILNN smoke test passed.")


if __name__ == "__main__":
    main()
