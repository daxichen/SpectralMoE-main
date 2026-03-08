import os
from PIL import Image

# 定义文件路径
image_dir = 'Your input image path'
label_dir = 'Your input label path'
output_image_dir = 'Your output image path'
output_label_dir = 'Your output label path'

# 创建输出目录
os.makedirs(output_image_dir, exist_ok=True)
os.makedirs(output_label_dir, exist_ok=True)

# 定义裁剪区域参数
left = 60
right_pad = 40
top = 54
bottom_pad = 54

# 原始图像尺寸
original_width = 7300  # 注意PIL的尺寸是（宽，高）
original_height = 6908

# 计算裁剪区域
crop_box = (
    left,  # 左边界
    top,   # 上边界
    original_width - right_pad,  # 右边界
    original_height - bottom_pad  # 下边界
)

# 处理所有图像文件
for filename in os.listdir(image_dir):
    if filename.endswith('.tif'):
        # 处理多光谱图像
        img_path = os.path.join(image_dir, filename)
        with Image.open(img_path) as img:
            cmyk_img = img.convert('CMYK')
            cropped_img = cmyk_img.crop(crop_box)
            
            # 保存裁剪后的图像
            output_path = os.path.join(output_image_dir, filename)
            cropped_img.save(output_path, format='TIFF')

        # 处理对应的标签文件
        base_name = filename.replace('.tif', '')
        label_filename = f"{base_name}_24label.png"
        label_path = os.path.join(label_dir, label_filename)
        label_save_filename = f"{base_name}.png"
        
        with Image.open(label_path) as label:
            cropped_label = label.crop(crop_box)
            
            # 保存裁剪后的标签
            output_label_path = os.path.join(output_label_dir, label_save_filename)
            cropped_label.save(output_label_path, format='PNG')

print("处理完成！所有文件已保存到指定目录。")