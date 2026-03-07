# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.registry import DATASETS
from mmseg.datasets import BaseSegDataset


@DATASETS.register_module()
class OpenEarthMapDataset(BaseSegDataset):
    """OpenEarthMap dataset.

    In segmentation map annotation for OpenEarthMap, 0 is the ignore index.
    ``reduce_zero_label`` should be set to True. The ``img_suffix`` and
    ``seg_map_suffix`` are both fixed to '.png'.
    """
    METAINFO = dict(
        classes=('bareland', 'rangeland', 'developed space', 'road', 'tree', 'water', 'agriculture land', 'building'),
        palette=[[128, 0, 0], [0, 255, 36], [148, 148, 148], [255, 255, 255],
                 [34, 97, 38], [0, 69, 255], [75, 181, 73], [222, 31, 7]]
        )
    # METAINFO = dict(
    #     classes=('rangeland', 'developed space', 'road', 'tree', 'water', 'agriculture land', 'building'),
    #     palette=[[0, 255, 36], [148, 148, 148], [255, 255, 255],
    #              [34, 97, 38], [0, 69, 255], [75, 181, 73], [222, 31, 7]]
    #     )

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
