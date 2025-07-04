#==============================================================================#  
#  Authors:     Xinyi Wang                                                     #
#  License:     GNU GPL v3.0                                                   #
#==============================================================================#
#-----------------------------------------------------#
#                   Library imports                   #
#-----------------------------------------------------#
# External libraries
import numpy as np
import torch
from typing import Optional, Union, Any

#-----------------------------------------------------#
#                     FODSample Class                 #
#-----------------------------------------------------#
class Sample:
    """
    A container class for a single sample of fiber orientation distribution (FOD)
    data, including image, segmentation, prediction, and associated metadata.

    This class supports multi-channel image data and optional annotations
    for segmentation or ground truth/prediction.
    """

    def __init__(
        self,
        index: Union[int, str],
        image: Union[np.ndarray, torch.Tensor],
        channels: int,
        classes: int,
        info: Optional[dict] = None,
    ):
        """
        Initialize a new FODSample instance.

        Args:
            index (int | str): Identifier of the sample.
            image (ndarray | Tensor): Input image data (e.g., FOD volume).
            channels (int): Number of image channels.
            classes (int): Number of segmentation or prediction classes.
            info (dict, optional): Additional metadata (e.g., subject ID, affine, site).
        """
        self.index: Union[int, str] = index
        self.img_data: Union[np.ndarray, torch.Tensor] = image
        self.channels: int = channels
        self.classes: int = classes
        self.shape: tuple = image.shape
        self.info: Optional[dict] = info

        # Optional annotations
        self.seg_data: Optional[Union[np.ndarray, torch.Tensor]] = None
        self.gt_data: Optional[Union[np.ndarray, torch.Tensor]] = None
        self.pred_data: Optional[Union[np.ndarray, torch.Tensor]] = None
        self.details: Optional[Any] = None

    def add_segmentation(self, seg: Union[np.ndarray, torch.Tensor]) -> None:
        """Attach a segmentation annotation to the sample."""
        self.seg_data = seg

    def add_gt(self, gt: Union[np.ndarray, torch.Tensor]) -> None:
        """Attach ground truth data to the sample."""
        self.gt_data = gt

    def add_prediction(self, pred: Union[np.ndarray, torch.Tensor]) -> None:
        """Attach model prediction to the sample."""
        self.pred_data = pred

    def add_details(self, details: Any) -> None:
        """Attach additional metadata or processing details."""
        self.details = details
    
    def summary(self) -> str:
        """Return a human-readable summary of the sample."""
        return (
            f"FODSample(index={self.index}, shape={self.shape}, "
            f"channels={self.channels}, classes={self.classes}, "
            f"has_seg={'Yes' if self.seg_data is not None else 'No'}, "
            f"has_gt={'Yes' if self.gt_data is not None else 'No'}, "
            f"has_pred={'Yes' if self.pred_data is not None else 'No'})"
        )

