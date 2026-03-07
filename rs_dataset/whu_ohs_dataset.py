from mmseg.registry import DATASETS
from mmseg.datasets import BaseSegDataset


@DATASETS.register_module()
class WHU_OHS_Dataset(BaseSegDataset):
    """WHU-OHS Dataset for Semantic Segmentation.

    Args:
        img_suffix (str): 图像文件的后缀名，默认为 `.tif`。
        seg_map_suffix (str): 标签文件的后缀名，默认为 `.tif`。
        reduce_zero_label (bool): 是否将标签中的 0 值视为忽略区域。
        **kwargs: 其他参数传递给 BaseSegDataset。
    """

    # METAINFO = dict(
    #     classes=(
    #         'background','paddy_field', 'dry_farm', 'woodland', 'shrubbery', 'sparse_woodland', 'other_forest_land', 'high-covered_grassland',
    #         'medium-covered_grassland', 'low-covered_grassland', 'river_canal', 'lake', 'reservoir_pond', 'beach land', 'shoal',
    #         'urban_built-up', 'rural_settlement', 'other_construction land', 'sand', 'gobi', 'saline-alkali_soil', 'marshland',
    #         'bare_land', 'bare_rock', 'ocean'
    #     ),
    #     palette=[
    #         [255, 255, 255], [190, 210, 255], [0, 255, 197], [38, 115, 0],
    #         [163, 255, 115], [76, 230, 0], [85, 255, 0], [115, 115, 0],
    #         [168, 168, 0], [255, 255, 0], [115, 178, 255], [0, 92, 230],
    #         [0, 38, 115], [122, 142, 245], [0, 168, 132], [115, 0, 0],
    #         [255, 127, 127], [255, 190, 190], [255, 190, 232], [255, 0, 197],
    #         [230, 0, 169], [168, 0, 132], [115, 0, 76], [255, 115, 223],
    #         [161, 161, 161]
    #     ]
    # )
    
    # METAINFO = dict(
    #     classes=(
    #         'paddy_field', 'dry_farm', 'woodland', 'shrubbery', 'sparse_woodland', 'other_forest_land', 'high-covered_grassland',
    #         'medium-covered_grassland', 'low-covered_grassland', 'river_canal', 'lake', 'reservoir_pond', 'beach land', 'shoal',
    #         'urban_built-up', 'rural_settlement', 'other_construction land', 'sand', 'gobi', 'saline-alkali_soil', 'marshland',
    #         'bare_land', 'bare_rock', 'ocean'
    #     ),
    #     palette=[
    #         [190, 210, 255], [0, 255, 197], [38, 115, 0],
    #         [163, 255, 115], [76, 230, 0], [85, 255, 0], [115, 115, 0],
    #         [168, 168, 0], [255, 255, 0], [115, 178, 255], [0, 92, 230],
    #         [0, 38, 115], [122, 142, 245], [0, 168, 132], [115, 0, 0],
    #         [255, 127, 127], [255, 190, 190], [255, 190, 232], [255, 0, 197],
    #         [230, 0, 169], [168, 0, 132], [115, 0, 76], [255, 115, 223],
    #         [161, 161, 161]
    #     ]
    # )
    
    METAINFO = dict(
        classes=(
            'Farmland', 'Forest', 'Grassland', 'Waterbody', 'Built-upland', 'Unusedland', 'Ocean'
        ),
        palette=[
            [0, 255, 197], [38, 115, 0],  [85, 255, 0],
            [0, 92, 230], [115, 0, 0], [255, 255, 0], [0, 38, 115]
        ]
    )
    
    # METAINFO = dict(
    #     classes=(
    #         'Farmland', 'Forest', 'Grassland', 'Waterbody', 'Built-upland', 'Ocean'
    #     ),
    #     palette=[
    #         [0, 255, 197], [38, 115, 0],  [85, 255, 0],
    #         [0, 92, 230], [115, 0, 0], [0, 38, 115]
    #     ]
    # )
    
    # METAINFO = dict(
    #     classes=(
    #         'Farmland', 'Forest', 'Grassland', 'Waterbody', 'Built-upland'
    #     ),
    #     palette=[
    #         [0, 255, 197], [38, 115, 0],  [85, 255, 0],
    #         [0, 92, 230], [115, 0, 0]
    #     ]
    # )

    def __init__(self,
                 img_suffix='.tif',
                 seg_map_suffix='.tif',
                 reduce_zero_label=True,
                 **kwargs):
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs
        )

