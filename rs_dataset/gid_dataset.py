from mmseg.registry import DATASETS
from mmseg.datasets import BaseSegDataset


@DATASETS.register_module()
class GID_Dataset(BaseSegDataset):
    """WHU-OHS Dataset for Semantic Segmentation.

    Args:
        img_suffix (str): 图像文件的后缀名，默认为 `.tif`。
        seg_map_suffix (str): 标签文件的后缀名，默认为 `.png`。
        reduce_zero_label (bool): 是否将标签中的 0 值视为忽略区域。
        **kwargs: 其他参数传递给 BaseSegDataset。
    """
    
    # METAINFO = dict(
    #     classes=('Built-up', 'Farmland', 'Forest', 'Meadow', 'Water'),
    #     palette=[[255, 0, 0], [0, 255, 0], [0, 255, 255],
    #              [255, 255, 0], [0, 0, 255]])
    
    
    METAINFO = dict(
        classes=(
            'industrial_area', 'paddy_field', 'irrigated_field', 'dry_cropland', 'garden_land', 'arbor_forest', 'shrub_forest',
            'park', 'natural_meadow', 'artificial_meadow', 'river', 'urban_residential', 'lake', 'pond',
            'fish_pond', 'snow', 'bare_land', 'rural_residential', 'stadium', 'square', 'road',
            'overpass', 'railway_station', 'airport'
        ),
        palette=[
            [200, 0, 0], [0, 200, 0], [150, 250, 0], [150, 200, 150],
            [200, 0, 200], [150, 0, 250], [150, 150, 250], [200, 150, 200],
            [250, 200, 0], [200, 200, 0], [0, 0, 200],
            [250, 0, 150], [0, 150, 200], [0, 200, 250], [150, 200, 250],
            [250, 250, 250], [200, 200, 200], [200, 150, 150], [250, 200, 150],
            [150, 150, 0], [250, 150, 150], [250, 150, 0], [250, 200, 250],
            [200, 150, 0]
        ]
    )

    def __init__(self,
                 img_suffix='.tif',
                 seg_map_suffix='.png',
                 reduce_zero_label=True,
                 **kwargs):
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs
        )

