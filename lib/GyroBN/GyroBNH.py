import torch
import torch.nn as nn

# from .Geometry.hyperbolic import Poincare,Hyperboloid
from .Geometry.constantcurvature.hyperboloid import Hyperboloid
Klein,Poincare = Hyperboloid,Hyperboloid
from .GyroBNBase import GyroBNBase
import share
class GyroBNH(GyroBNBase):
    """
    GyroBN layer for hyperbolic spaces (Poincare, Hyperboloid, or Klein models).

    Note: This implementation only supports input of shape [batch_size, dim].

    Args:
        shape (list[int]): Manifold dimensions, e.g., [d] for vectors in L^d, which are represented as vectors in R^{d+1}.
        model (str, default='Poincare'): Type of hyperbolic model. One of {"Poincare", "Hyperboloid", "Klein"}.
        K (float, default=-1.0): Negative curvature of the hyperbolic space.
        max_iter (int, default=1000): Maximum iterations for computing the Fréchet mean.
    """

    def __init__(self, shape, model='Poincare', K=-1.0, max_iter=1000, momentum=0.1, translate="Left",
                 track_running_stats=None, init_1st_batch=False, eps=1e-6):
        if track_running_stats is None:
            track_running_stats = share.share_track_running_stats
        super().__init__(shape=shape, batchdim=[0], momentum=momentum, translate=translate,
                         track_running_stats=track_running_stats, init_1st_batch=init_1st_batch, eps=eps)
        self.model = model
        self.K = K
        self.max_iter = max_iter if max_iter is not None else share.share_max_iter
        # print("GyroBNH max_iter: ", self.max_iter)
        self.get_manifold()
        self.set_parameters()
        if self.track_running_stats:
            self.register_running_stats()
        

    def get_manifold(self):
        classes = {
            "Poincare": Poincare,
            "Hyperboloid": Hyperboloid,
            "Klein": Klein,
        }
        self.manifold = classes[self.model](K=self.K)

    def set_parameters(self):
        mean_shape, var_shape = self._get_param_shapes()
        self.weight = nn.Parameter(self.manifold.zero_tan(mean_shape))
        self.shift = nn.Parameter(torch.ones(*var_shape)*0.5)

    def register_running_stats(self):
        """
        In [a], they use the following for the variance:
            Initialization:
                self.updates = 0
            updates:
                self.running_var = (1 - 1 / self.updates) * self.running_var + batch_var / self.updates
                self.updates += 1
        But we follow the standard Euclidean BN.
        [a] Differentiating through the Fréchet Mean
        """
        if self.init_1st_batch:
            self.running_mean = None
            self.running_var = None
        else:
            mean_shape, var_shape = self._get_param_shapes()
            running_mean = self.manifold.zero(mean_shape)
            running_var = torch.ones(*var_shape)
            self.register_buffer('running_mean', running_mean)
            self.register_buffer('running_var', running_var)
    def forward_(self, x):
        
        # print("GyroBNH x is finate? ->", torch.isfinite(x).all(), "max|x| =", x.abs().amax())
        # print("self.weight: ", self.weight)
        # print("self.shift: ", self.shift)
        # print("self.training: ", self.training)
        # print("self.max_iter: ", self.max_iter)
        # print("self.track_running_stats: ", self.track_running_stats)
        if self.training:
            input_mean = self.manifold.frechet_mean(x,max_iter=self.max_iter)
            input_var = self.manifold.frechet_variance(x, input_mean)
            if self.track_running_stats:
                self.updating_running_statistics(input_mean, input_var)

            # print("GyroBNH input_mean is finate? ->", torch.isfinite(input_mean).all(), "max|input_mean| =", input_mean.abs().amax())
            # print("GyroBNH input_var is finate? ->", torch.isfinite(input_var).all(), "max|input_var| =", input_var.abs().amax())
        elif self.track_running_stats:
            input_mean = self.running_mean
            input_var = self.running_var
            # print("GyroBNH track_running_stats input_mean is finate? ->", torch.isfinite(input_mean).all(), "max|input_mean| =", input_mean.abs().amax())
            # print("GyroBNH track_running_stats input_var is finate? ->", torch.isfinite(input_var).all(), "max|input_var| =", input_var.abs().amax())
        else:
            input_mean = self.manifold.frechet_mean(x,max_iter=self.max_iter)
            input_var = self.manifold.frechet_variance(x, input_mean)
            # print("GyroBNH else input_mean is finate? ->", torch.isfinite(input_mean).all(), "max|input_mean| =", input_mean.abs().amax())
            # print("GyroBNH else input_var is finate? ->", torch.isfinite(input_var).all(), "max|input_var| =", input_var.abs().amax())


        output = self.normalization(x,input_mean,input_var)
        # print("GyroBNH output is finate? ->", torch.isfinite(output).all(), "max|output| =", output.abs().amax())
        return output
    def forward(self, x):
        
        # print("GyroBNH x is finate? ->", torch.isfinite(x).all(), "max|x| =", x.abs().amax())
        # print("GyroBNH track_running_stats: ", self.track_running_stats)
        # with open("shift.txt", "a+") as f:
            # f.write(str(self.shift) + "\n")
        if self.training:
            # input_mean = self.manifold.frechet_mean(x,max_iter=self.max_iter)
            # input_var = self.manifold.frechet_variance(x, input_mean)

            """wai yun"""
            input_mean = share.share_manifold.centroid(x)
            x_T = share.share_manifold.logmap(input_mean, x)
            x_T0 = share.share_manifold.transp0back(input_mean, x_T)
            input_var = torch.mean(torch.norm(x_T0, dim=-1), dim=0)
            

            if self.track_running_stats:
                self.updating_running_statistics(input_mean, input_var)

            # print("GyroBNH input_mean is finate? ->", torch.isfinite(input_mean).all(), "max|input_mean| =", input_mean.abs().amax())
            # print("GyroBNH input_var is finate? ->", torch.isfinite(input_var).all(), "max|input_var| =", input_var.abs().amax())
        elif self.track_running_stats:
            input_mean = self.running_mean
            input_var = self.running_var
        else:
            # input_mean = self.manifold.frechet_mean(x,max_iter=self.max_iter)
            # input_var = self.manifold.frechet_variance(x, input_mean) 
            input_mean = share.share_manifold.centroid(x)
            x_T = share.share_manifold.logmap(input_mean, x)
            x_T0 = share.share_manifold.transp0back(input_mean, x_T)
            input_var = torch.mean(torch.norm(x_T0, dim=-1), dim=0)



        output = self.normalization(x,input_mean,input_var)
        # print("GyroBNH output is finate? ->", torch.isfinite(output).all(), "max|output| =", output.abs().amax())
        return output

    def normalization(self,x,input_mean,input_var):
        # set parameters
        weight = self.manifold.exp0(self.weight)
        # centering
        inv_input_mean = self.manifold.gyroinv(input_mean)
        x_center = self.manifold.gyrotrans(inv_input_mean,x,translate=self.translate)
        # print("GyroBNH x_center is finate? ->", torch.isfinite(x_center).all(), "max|x_center| =", x_center.abs().amax())
        # shifting
        shift_clamp = torch.clamp(self.shift, min=-1.5, max=1.5)
        factor = torch.clamp(self.shift, min=-2.5, max=2.0) / (input_var + self.eps).sqrt()
        # factor = self.shift / (input_var + self.eps).sqrt()
        # factor = torch.clamp(factor, min=-2.0, max=2.0)
        # print("GyroBNH factor is finate? ->", torch.isfinite(factor).all(), "max|factor| =", factor.abs().amax())
        x_scaled = self.manifold.gyroscalarprod(x_center,factor)
        # print("GyroBNH x_scaled is finate? ->", torch.isfinite(x_scaled).all(), "max|x_scaled| =", x_scaled.abs().amax())
        # biasing
        x_normed = self.manifold.gyrotrans(weight, x_scaled,translate=self.translate)
        #   print("GyroBNH x_normed is finate? ->", torch.isfinite(x_normed).all(), "max|x_normed| =", x_normed.abs().amax())
        if not x_normed.isfinite().all():
            raise RuntimeError("GyroBNH x_normed is not finate")
        return x_normed
