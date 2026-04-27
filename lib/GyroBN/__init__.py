"""
@author: Ziheng Chen
Please cite the papers below if you use the code:

Ziheng Chen, Yue Song, Xiao-Jun Wu, and Nicu Sebe. Gyrogroup Batch Normalization. ICLR 2025.
Ziheng Chen, Yue Song, Xiao-Jun Wu, and Nicu Sebe. Batch Normalization over Manifolds: A Gyro Approach.

And also the following w.r.t. the special LieBN cases:

Ziheng Chen, Yue Song, Yunmei Liu, and Nicu Sebe. A Lie Group Approach to Riemannian Batch Normalization. ICLR 2024.
Ziheng Chen, Yue Song, Tianyang Xu, Zhiwu Huang, Xiao-Jun Wu, and Nicu Sebe. Adaptive Log-Euclidean metrics for SPD matrix learning. TIP 2024.
Ziheng Chen, Yue Song, Rui Wang, Xiao-Jun Wu, and Nicu Sebe. LieBN: Batch Normalization over Lie Groups.

Copyright (C) 2025 Ziheng Chen
All rights reserved.
"""

from .GyroBNBase import GyroBNBase