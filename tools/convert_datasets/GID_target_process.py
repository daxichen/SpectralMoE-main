import os
import numpy as np
from PIL import Image
from tqdm import tqdm

def batch_crop_to_annotation(input_img_dir, input_label_dir, output_img_dir, output_label_dir):
    """
    处理文件名完全相同的图像和标签对
    参数：
    input_img_dir: 输入图像文件夹路径（示例：D:/.../image）
    input_label_dir: 输入标签文件夹路径（示例：D:/.../label）
    output_img_dir: 输出裁剪图像路径
    output_label_dir: 输出裁剪标签路径
    """
    # 创建输出目录
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)

    # 获取所有匹配的文件对
    label_files = [f for f in os.listdir(input_label_dir) if f.endswith(('.tif', '.png'))]
    img_files = [f for f in os.listdir(input_img_dir) if f.endswith(('.tif', '.png'))]
    
    # 找到共有的文件名（根据实际需求选择匹配方式）
    common_files = set(label_files) & set(img_files)
    
    # 添加进度条
    for filename in tqdm(common_files, desc="Processing files"):
        try:
            # 构建完整路径
            label_path = os.path.join(input_label_dir, filename)
            image_path = os.path.join(input_img_dir, filename)
            
            # 生成输出路径（保持相同文件名）
            output_img_path = os.path.join(output_img_dir, f"crop_{filename}")
            output_label_path = os.path.join(output_label_dir, f"crop_{filename.replace('.tif', '.png')}")

            # 执行裁剪操作
            with Image.open(label_path) as lbl:
                label = lbl.convert('L')
                label_array = np.array(label)
                y_indices, x_indices = np.where(label_array > 0)
                
                if len(y_indices) == 0:
                    print(f"\nWarning: {filename} 中没有标注区域")
                    continue

                # 计算裁剪区域
                min_y, max_y = np.min(y_indices), np.max(y_indices)
                min_x, max_x = np.min(x_indices), np.max(x_indices)
                center_y = (min_y + max_y) // 2
                center_x = (min_x + max_x) // 2
                img_height, img_width = label_array.shape

                # 动态计算裁剪尺寸
                size = 1024
                half_size = size // 2

                # 自动调整边界
                start_y = max(0, center_y - half_size)
                end_y = min(img_height, start_y + size)
                start_y = max(0, end_y - size)

                start_x = max(0, center_x - half_size)
                end_x = min(img_width, start_x + size)
                start_x = max(0, end_x - size)

                # 转换为整数坐标
                start_x, start_y, end_x, end_y = map(int, (start_x, start_y, end_x, end_y))

            # 裁剪并保存图像
            with Image.open(image_path) as img:
                # 保留原始模式（CMYK/RGB等）
                cropped_img = img.crop((start_x, start_y, end_x, end_y))
                if filename.endswith('.tif'):
                    cropped_img.save(output_img_path, format="TIFF", compression="tiff_deflate")
                else:
                    cropped_img.save(output_img_path)

            # 裁剪并保存标签
            with Image.open(label_path) as lbl:
                cropped_lbl = lbl.crop((start_x, start_y, end_x, end_y))
                cropped_lbl.save(output_label_path, format="PNG")

        except Exception as e:
            print(f"\n处理文件 {filename} 时发生错误：{str(e)}")

if __name__ == "__main__":
    # 输入输出路径配置
    input_img_dir = r"Your input image path"
    input_label_dir = r"Your input label path"
    output_img_dir = r"Your output image path"
    output_label_dir = r"Your output label path"

    # 执行批量处理
    batch_crop_to_annotation(input_img_dir, input_label_dir, output_img_dir, output_label_dir)