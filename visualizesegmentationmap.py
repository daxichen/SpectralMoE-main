import os
import numpy as np
from PIL import Image
import tifffile 

# 路径配置

pred_dir = 'work_dirs/SpectralMoE_potsdam2vaihingen'
label_dir = 'data/Potsdam2Vaihingen/Vaihingen/label'
output_dir = 'work_dirs/SpectralMoE_potsdam2vaihingen_RGB'

# 创建输出目录
os.makedirs(output_dir, exist_ok=True)

# WHUOHS
# 颜色映射表（0-6对应7个类别）
# color_map = {
#     0: [255, 255, 255],   # 白色背景
#     1: [0, 255, 197],     # 青绿色
#     2: [38, 115, 0],      # 深绿色
#     3: [85, 255, 0],      # 亮绿色
#     4: [0, 92, 230],      # 蓝色
#     5: [115, 0, 0],       # 深红色
#     6: [255, 255, 0],     # 黄色
#     7: [0, 38, 115]       # 深蓝色
# }

# color_map = {
#     0: [0, 0, 0],          # 黑色背景
#     1: [200, 0, 0],        # 深红
#     2: [0, 200, 0],        # 亮绿
#     3: [150, 250, 0],      # 黄绿
#     4: [150, 200, 150],    # 灰绿
#     5: [200, 0, 200],      # 品红
#     6: [150, 0, 250],      # 蓝紫
#     7: [150, 150, 250],    # 淡蓝
#     8: [200, 150, 200],    # 粉紫
#     9: [250, 200, 0],      # 橙黄
#     10: [200, 200, 0],     # 黄
#     11: [0, 0, 200],       # 深蓝
#     12: [250, 0, 150],     # 玫红
#     13: [0, 150, 200],     # 湖蓝
#     14: [0, 200, 250],     # 天蓝
#     15: [150, 200, 250],   # 浅天蓝
#     16: [250, 250, 250],   # 纯白
#     17: [200, 200, 200],   # 浅灰
#     18: [200, 150, 150],   # 粉红
#     19: [250, 200, 150],   # 肉色
#     20: [150, 150, 0],     # 橄榄绿
#     21: [250, 150, 150],   # 浅粉
#     22: [250, 150, 0],     # 橙色
#     23: [250, 200, 250],   # 浅粉紫
#     24: [200, 150, 0]      # 深橙
# }

# LoveDA
# color_map = {
#     0: [0, 0, 0],
#     1: [255, 255, 255],
#     2: [255, 0, 0],
#     3: [255, 255, 0],
#     4: [0, 0, 255],
#     5: [159, 129, 183],
#     6: [0, 255, 0],
#     7: [255, 195, 128]
# }

# Potsdam
color_map = {
    0: [0, 0, 0],
    1: [255, 255, 255],    
    2: [0, 0, 255],    
    3: [0, 255, 255],     
    4: [0, 255, 0],    
    5: [255, 255, 0],      
    6: [255, 0, 0]
}

# OpenEarthMap
# color_map = {
#     0: [0, 0, 0],
#     1: [128, 0, 0],
#     2: [0, 255, 36],
#     3: [148, 148, 148],
#     4: [255, 255, 255],
#     5: [34, 97, 38],
#     6: [0, 69, 255],
#     7: [75, 181, 73],
#     8: [222, 31, 7],
# }

# FLAIR
# color_map = {
#     0: [0, 0, 0],
#     1: [219, 14, 154],
#     2: [147, 142, 123],
#     3: [248, 12, 0],
#     4: [169, 113, 1],
#     5: [21, 83, 174],
#     6: [25, 74, 38],
#     7: [70, 228, 131],
#     8: [243, 166, 13],
#     9: [102, 0, 130],
#     10: [85, 255, 0],
#     11: [255, 243, 13],
#     12: [228, 223, 124],
# }

# 处理每个预测结果
for pred_file in os.listdir(pred_dir):
    if not pred_file.endswith('.png'):
        continue

    # 构建对应文件路径
    base_name = os.path.splitext(pred_file)[0]
    pred_path = os.path.join(pred_dir, pred_file)
    label_path = os.path.join(label_dir, f'{base_name}.tif')  # 假设文件名一致
    # label_path = os.path.join(label_dir, f'{base_name}.png')  # 假设文件名一致

    # 读取数据
    pred = np.array(Image.open(pred_path))  # 预测结果（1-6）
    # label = tifffile.imread(label_path)     # 真实标签（含0背景）
    label = Image.open(label_path)     # 真实标签（含0背景）
    label= np.array(label, dtype=np.uint8)

    # 确保尺寸一致
    if pred.shape != label.shape:
        raise ValueError(f"尺寸不匹配: {pred_file} {pred.shape} vs {label.shape}")

    # 用真实标签的0替换预测结果
    merged = pred.copy()
    merged[label == 0] = 0  # 将真实标签中0的位置设为0

    # 创建RGB图像
    rgb = np.zeros((*merged.shape, 3), dtype=np.uint8)
    for class_id, color in color_map.items():
        rgb[merged == class_id] = color

    # 保存结果
    output_path = os.path.join(output_dir, pred_file)
    Image.fromarray(rgb).save(
        output_path,
        dpi=(600, 600),        # 设置分辨率
        format='PNG',
        compress_level=0        # 禁用压缩以保证最高质量
    )

print(f"处理完成！结果保存在：{output_dir}")
