# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from typing import Dict, Optional, Union
from mmseg.registry import TRANSFORMS
import numpy as np
from sklearn.decomposition import PCA
from mmcv.transforms import BaseTransform
from mmcv.transforms import LoadAnnotations as MMCV_LoadAnnotations
try:
    from osgeo import gdal
    from PIL import Image
except ImportError:
    gdal = None
    Image = None


@TRANSFORMS.register_module()
class LoadAnnotationsForTIF(MMCV_LoadAnnotations):
    """Load annotations for semantic segmentation with GDAL for `.tif` files.

    This class extends the `LoadAnnotations` and adds support for loading
    `.tif` labels using GDAL.

    Args:
        reduce_zero_label (bool, optional): Whether to reduce all label values
            by 1. Defaults to None.
        backend_args (dict): Arguments to instantiate file backend.
            Defaults to None.
        imdecode_backend (str): Backend for decoding other image formats.
            Defaults to 'pillow'.
        use_gdal (bool): Whether to use GDAL to load `.tif` files.
            Defaults to True.
    """

    def __init__(
        self,
        reduce_zero_label=None,
        backend_args=None,
        imdecode_backend='pillow'
    ) -> None:
        super().__init__(
            with_bbox=False,
            with_label=False,
            with_seg=True,
            with_keypoints=False,
            imdecode_backend=imdecode_backend,
            backend_args=backend_args
        )
        self.reduce_zero_label = reduce_zero_label

        if self.reduce_zero_label is not None:
            warnings.warn(
                '`reduce_zero_label` will be deprecated in future versions. '
                'If you want to ignore the zero label, please set '
                '`reduce_zero_label=True` when initializing the dataset.'
            )
        self.imdecode_backend = imdecode_backend

    def _load_seg_map(self, results: dict) -> None:
        """Private function to load semantic segmentation annotations.

        Args:
            results (dict): Result dict from :obj:``mmcv.BaseDataset``.

        Returns:
            dict: The dict contains loaded semantic segmentation annotations.
        """

        # 使用 GDAL 加载 .tif 文件
        seg_map_path = results['seg_map_path']
        dataset = gdal.Open(seg_map_path, gdal.GA_ReadOnly)
        if dataset is None:
            raise RuntimeError(f'Cannot open segmentation map file: {seg_map_path}')
        
        # 读取为 NumPy 数组
        gt_semantic_seg = dataset.ReadAsArray().astype(np.uint8)

        # reduce zero label
        if self.reduce_zero_label is None:
            self.reduce_zero_label = results['reduce_zero_label']
        assert self.reduce_zero_label == results['reduce_zero_label'], \
            'Initialize dataset with `reduce_zero_label` as ' \
            f'{results["reduce_zero_label"]} but when loading annotation ' \
            f'the `reduce_zero_label` is {self.reduce_zero_label}'
        if self.reduce_zero_label:
            # 避免使用溢出转换
            gt_semantic_seg[gt_semantic_seg == 0] = 255
            gt_semantic_seg = gt_semantic_seg - 1
            gt_semantic_seg[gt_semantic_seg == 254] = 255

        # 如果有 label_map，映射标签
        if results.get('label_map', None) is not None:
            gt_semantic_seg_copy = gt_semantic_seg.copy()
            for old_id, new_id in results['label_map'].items():
                gt_semantic_seg[gt_semantic_seg_copy == old_id] = new_id

        # 更新 results
        results['gt_seg_map'] = gt_semantic_seg
        results['seg_fields'].append('gt_seg_map')
        
    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(reduce_zero_label={self.reduce_zero_label}, '
        repr_str += f"imdecode_backend='{self.imdecode_backend}', "
        repr_str += f'backend_args={self.backend_args})'
        return repr_str
    
# @TRANSFORMS.register_module()
# class LoadHyperRSImageFromFile(BaseTransform):
#     def __init__(self, to_float32: bool = True, rgb_bands: list = [13, 5, 0]):
#         self.to_float32 = to_float32
#         self.rgb_bands = rgb_bands  

#         if gdal is None:
#             raise RuntimeError('gdal is not installed')

#     def transform(self, results: Dict) -> Dict:
#         filename = results['img_path']
#         ds = gdal.Open(filename)
#         if ds is None:
#             raise Exception(f'Unable to open file: {filename}')
        
#         img_all_bands = ds.ReadAsArray()
        
#         rgb_data = img_all_bands[self.rgb_bands, :, :]  # 形状 (3, height, width)
#         rgb_image = np.einsum('ijk->jki', rgb_data)
        
#         rgb_image = rgb_image / 10000.0
        
#         if self.to_float32:
#             rgb_image = rgb_image.astype(np.float32)
        
#         results['img'] = rgb_image
#         results['img_shape'] = rgb_image.shape[:2]
#         results['ori_shape'] = rgb_image.shape[:2]
        
#         return results

@TRANSFORMS.register_module()
class LoadHyperRSImageFromFile(BaseTransform):
    def __init__(self, to_float32: bool = True):
        self.to_float32 = to_float32

        if gdal is None:
            raise RuntimeError('gdal is not installed')

    def transform(self, results: Dict) -> Dict:
        filename = results['img_path']
        ds = gdal.Open(filename)
        if ds is None:
            raise Exception(f'Unable to open file: {filename}')
        
        img_all_bands = ds.ReadAsArray()
        
        rgb_image = np.einsum('ijk->jki', img_all_bands)
        # rgb_image = rgb_image[:, :, ::3][:, :, :10]
        rgb_image = rgb_image / 10000.0
        
        if self.to_float32:
            rgb_image = rgb_image.astype(np.float32)
        
        results['img'] = rgb_image
        results['img_shape'] = rgb_image.shape[:2]
        results['ori_shape'] = rgb_image.shape[:2]
        
        return results
    
# @TRANSFORMS.register_module()
# class LoadHyperRSImageFromFile(BaseTransform):
#     """Load a Remote Sensing mage from file.

#     Required Keys:

#     - img_path

#     Modified Keys:

#     - img
#     - img_shape
#     - ori_shape

#     Args:
#         to_float32 (bool): Whether to convert the loaded image to a float32
#             numpy array. If set to False, the loaded image is a float64 array.
#             Defaults to True.
#     """

#     def __init__(self, to_float32: bool = True, pca_components: int = 3):
#         self.to_float32 = to_float32
#         self.pca_components = pca_components

#         if gdal is None:
#             raise RuntimeError('gdal is not installed')

#     def transform(self, results: Dict) -> Dict:
#         """Functions to load image.

#         Args:
#             results (dict): Result dict from :obj:``mmcv.BaseDataset``.

#         Returns:
#             dict: The dict contains loaded image and meta information.
#         """

#         filename = results['img_path']
#         ds = gdal.Open(filename)
#         if ds is None:
#             raise Exception(f'Unable to open file: {filename}')
#         # img = np.einsum('ijk->jki', ds.ReadAsArray())
#         img = ds.ReadAsArray()
        
#         height, width, bands = img.shape[1], img.shape[2], img.shape[0]
#         img_reshaped = img.reshape(bands, -1).transpose(1, 0)
#         pca = PCA(n_components=self.pca_components)
#         img_pca = pca.fit_transform(img_reshaped)
#         img_pca = img_pca.reshape(height, width, self.pca_components)

#         if self.to_float32:
#             # img = img.astype(np.float32)
#             img_pca = img_pca.astype(np.float32)

#         results['img'] = img_pca
#         results['img_shape'] = img_pca.shape[:2]
#         results['ori_shape'] = img_pca.shape[:2]

#         return results

#     def __repr__(self):
#         repr_str = (f'{self.__class__.__name__}('
#                     f'to_float32={self.to_float32}, '
#                     f'pca_components={self.pca_components})')
#         return repr_str
    
@TRANSFORMS.register_module()
class LoadAnnotationsWithPIL(MMCV_LoadAnnotations):
    """Load annotations for semantic segmentation using PIL for image files.

    Args:
        reduce_zero_label (bool, optional): Whether to reduce all label values
            by 1. Defaults to None.
        backend_args (dict): Arguments to instantiate file backend.
            Defaults to None.
        imdecode_backend (str): Backend for decoding other image formats.
            Defaults to 'pillow'.
    """

    def __init__(
        self,
        reduce_zero_label=None,
        backend_args=None,
        imdecode_backend='pillow'
    ) -> None:
        super().__init__(
            with_bbox=False,
            with_label=False,
            with_seg=True,
            with_keypoints=False,
            imdecode_backend=imdecode_backend,
            backend_args=backend_args
        )
        self.reduce_zero_label = reduce_zero_label

        if self.reduce_zero_label is not None:
            warnings.warn(
                '`reduce_zero_label` will be deprecated in future versions. '
                'If you want to ignore the zero label, please set '
                '`reduce_zero_label=True` when initializing the dataset.'
            )

    def _load_seg_map(self, results: dict) -> None:
        """Load semantic segmentation annotations with PIL.

        Args:
            results (dict): Result dict from dataset.
        """
        from PIL import Image
        
        seg_map_path = results['seg_map_path']
        
        try:
            # 使用PIL打开图像文件
            pil_img = Image.open(seg_map_path)
            gt_semantic_seg = np.array(pil_img, dtype=np.uint8)
            
            # 处理灰度图像维度
            if len(gt_semantic_seg.shape) == 3:
                gt_semantic_seg = gt_semantic_seg.squeeze(axis=-1)
                
        except Exception as e:
            raise RuntimeError(f'Error loading segmentation map {seg_map_path}: {e}')

        # 处理零标签
        if self.reduce_zero_label is None:
            self.reduce_zero_label = results.get('reduce_zero_label', False)
            
        if self.reduce_zero_label:
            gt_semantic_seg = self.reduce_zero_label_transform(gt_semantic_seg)

        # 标签映射
        if results.get('label_map', None) is not None:
            gt_semantic_seg = self.apply_label_map(gt_semantic_seg, results['label_map'])

        results['gt_seg_map'] = gt_semantic_seg
        results['seg_fields'].append('gt_seg_map')

    def reduce_zero_label_transform(self, seg_map: np.ndarray) -> np.ndarray:
        """处理零标签转换"""
        temp_seg = seg_map.copy()
        temp_seg[temp_seg == 0] = 255
        temp_seg -= 1
        temp_seg[temp_seg == 254] = 255
        return temp_seg

    def apply_label_map(self, seg_map: np.ndarray, label_map: dict) -> np.ndarray:
        """应用标签映射"""
        new_seg = seg_map.copy()
        for old_id, new_id in label_map.items():
            new_seg[seg_map == old_id] = new_id
        return new_seg

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(reduce_zero_label={self.reduce_zero_label}, '
        repr_str += f"imdecode_backend='{self.imdecode_backend}', "
        repr_str += f'backend_args={self.backend_args})'
        return repr_str
    
@TRANSFORMS.register_module()
class LoadMultiSpectralImageFromFile(BaseTransform):
    """Load a Multi-Spectral Image from file using PIL.

    Required Keys:

    - img_path

    Modified Keys:

    - img
    - img_shape
    - ori_shape

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image will be a uint8 array.
            Defaults to True.
    """

    def __init__(self, to_float32: bool = True):
        self.to_float32 = to_float32

    def transform(self, results: Dict) -> Dict:
        """Functions to load multi-spectral image.

        Args:
            results (dict): Result dict from :obj:``mmcv.BaseDataset``.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        from PIL import Image  # Local import to avoid dependency issues

        filename = results['img_path']
        img = np.array(Image.open(filename).convert('CMYK'))  # Convert to CMYK and get numpy array
        
        if self.to_float32:
            img = img.astype(np.float32)

        results['img'] = img
        results['img_shape'] = img.shape[:2]
        results['ori_shape'] = img.shape[:2]
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'to_float32={self.to_float32})')
        return repr_str