from lib.lorentz.manifold import CustomLorentz

# Global switches used by the Lorentz modules. The training entry points set
# these values from their command-line arguments before constructing models.
share_bn_new = 1
share_FC_new = 1
share_b = 1

share_max_iter = 1000
share_dropout_rate = 0.0
share_dropout_weight = 0.0
share_track_running_stats = True

# GyroLBN uses a Lorentz centroid for its batch statistics. The paper fixes
# curvature to K=-1, represented here by geoopt's positive k=1.
share_manifold = CustomLorentz(k=1.0)

