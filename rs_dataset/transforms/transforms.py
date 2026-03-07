import copy
import inspect
import warnings
from typing import Dict, List, Optional, Sequence, Tuple, Union

import cv2
import mmcv
import mmengine
import numpy as np
from mmcv.transforms import RandomFlip as MMCV_RandomFlip
from mmcv.transforms import Resize as MMCV_Resize
from mmcv.transforms.base import BaseTransform
from mmcv.transforms.utils import cache_randomness
from mmengine.utils import is_tuple_of
from numpy import random
from scipy.ndimage import gaussian_filter

from mmseg.datasets.dataset_wrappers import MultiImageMixDataset
from mmseg.registry import TRANSFORMS

try:
    import albumentations
    from albumentations import Compose
    ALBU_INSTALLED = True
except ImportError:
    albumentations = None
    Compose = None
    ALBU_INSTALLED = False


@TRANSFORMS.register_module()
class MSIPhotoMetricDistortion(BaseTransform):
    """适用于四通道多光谱图像的光度变形（Nir+RGB）
    
    修改要点：
    1. 保持Nir通道不变，仅对RGB通道进行颜色相关变换
    2. 调整颜色空间转换逻辑以处理RGB三个通道
    3. 各变换操作仅应用于后三个通道（RGB）

    Required Keys:
    - img (四通道图像，形状为HxWx4)

    Modified Keys:
    - img
    """

    def __init__(self,
                 brightness_delta: int = 32,
                 contrast_range: Sequence[float] = (0.5, 1.5),
                 saturation_range: Sequence[float] = (0.5, 1.5),
                 hue_delta: int = 18):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    def convert(self, img: np.ndarray, alpha: float = 1.0, beta: float = 0.0) -> np.ndarray:
        """仅对RGB通道进行对比度/亮度调整"""
        # 分离Nir和RGB通道
        nir = img[..., 0:1]  # 保持维度
        rgb = img[..., 1:4]
        
        # 仅对RGB通道进行处理
        rgb = rgb.astype(np.float32) * alpha + beta
        rgb = np.clip(rgb, 0, 255)
        
        # 合并通道
        return np.concatenate([nir, rgb.astype(np.uint8)], axis=-1)

    def brightness(self, img: np.ndarray) -> np.ndarray:
        """仅调整RGB通道的亮度"""
        if random.randint(2):
            delta = random.uniform(-self.brightness_delta, self.brightness_delta)
            return self.convert(img, beta=delta)
        return img

    def contrast(self, img: np.ndarray) -> np.ndarray:
        """仅调整RGB通道的对比度"""
        if random.randint(2):
            alpha = random.uniform(self.contrast_lower, self.contrast_upper)
            return self.convert(img, alpha=alpha)
        return img

    def saturation(self, img: np.ndarray) -> np.ndarray:
        """调整饱和度（仅RGB通道）"""
        if random.randint(2):
            # 分离通道
            nir = img[..., 0]
            rgb = img[..., 1:4]
            
            # 转换到HSV空间
            hsv = mmcv.bgr2hsv(rgb[..., ::-1])  # RGB转BGR后再转HSV
            
            # 调整饱和度
            alpha = random.uniform(self.saturation_lower, self.saturation_upper)
            hsv[..., 1] = np.clip(hsv[..., 1].astype(np.float32) * alpha, 0, 255)
            
            # 转换回RGB
            rgb = mmcv.hsv2bgr(hsv)[..., ::-1]  # 转回RGB顺序
            
            return np.dstack([nir, rgb])
        return img

    def hue(self, img: np.ndarray) -> np.ndarray:
        """调整色调（仅RGB通道）"""
        if random.randint(2):
            # 分离通道
            nir = img[..., 0]
            rgb = img[..., 1:4]
            
            # 转换到HSV空间
            hsv = mmcv.bgr2hsv(rgb[..., ::-1])  # RGB转BGR后再转HSV
            
            # 调整色调
            delta = random.randint(-self.hue_delta, self.hue_delta)
            hsv[..., 0] = (hsv[..., 0].astype(int) + delta) % 180
            
            # 转换回RGB
            rgb = mmcv.hsv2bgr(hsv)[..., ::-1]  # 转回RGB顺序
            
            return np.dstack([nir, rgb])
        return img

    def transform(self, results: dict) -> dict:
        img = results['img']
        assert img.shape[2] == 4, "输入应为四通道图像"
        
        # 随机亮度调整
        img = self.brightness(img)
        
        # 对比度应用模式
        mode = random.randint(2)
        if mode == 1:
            img = self.contrast(img)
        
        # 饱和度调整
        img = self.saturation(img)
        
        # 色调调整
        img = self.hue(img)
        
        if mode == 0:
            img = self.contrast(img)
        
        results['img'] = img.astype(np.uint8)
        return results

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'brightness_delta={self.brightness_delta}, '
                f'contrast_range=({self.contrast_lower}, {self.contrast_upper}), '
                f'saturation_range=({self.saturation_lower}, {self.saturation_upper}), '
                f'hue_delta={self.hue_delta})')