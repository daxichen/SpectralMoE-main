# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.registry import DATASETS
from mmseg.datasets import BaseSegDataset


@DATASETS.register_module()
class FLAIRDataset(BaseSegDataset):
    """FLAIR dataset.

    In segmentation map annotation for FLAIR, 0 is the ignore index.
    ``reduce_zero_label`` should be set to True. The ``img_suffix`` and
    ``seg_map_suffix`` are both fixed to '.png'.
    """
    METAINFO = dict(
        classes=('Building', 'Pervious surface', 'Impervious surface', 'Bare soil', 'Water', 'Coniferous', 'Deciduous', 'Brushwood', 'Vineyard', 'Herbaceous vegetation', 'Agricultural land', 'Plowed land'),
        palette=[[219, 14, 154], [147, 142, 123], [248, 12,  0], [169, 113,  1], [21, 83, 174], [25, 74, 38],
                 [70, 228, 131], [243, 166, 13], [102, 0, 130], [ 85, 255,  0], [255, 243, 13], [228, 223,124]]
        )

    def __init__(self,
                 img_suffix='.tif',
                 seg_map_suffix='.tif',
                 reduce_zero_label=True,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)
