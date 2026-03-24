from gwf.losses.task_loss        import TaskLoss
from gwf.losses.smooth_loss      import SpatialSmoothnessLoss
from gwf.losses.kl_loss          import KLLoss
from gwf.losses.contrastive_loss import GeographicContrastiveLoss
from gwf.losses.gwf_loss         import GWFLoss

__all__ = [
    "TaskLoss",
    "SpatialSmoothnessLoss",
    "KLLoss",
    "GeographicContrastiveLoss",
    "GWFLoss",
]
