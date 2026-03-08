import os
from PIL import Image
import math

def crop_and_save_label_tiles(input_dir, output_dir):
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 遍历输入目录中的所有PNG文件
    for filename in os.listdir(input_dir):
        if not filename.lower().endswith('.png'):
            continue

        input_path = os.path.join(input_dir, filename)
        
        # 使用with语句确保正确关闭文件
        with Image.open(input_path) as img:
            # 处理调色板图像
            if img.mode == 'P':
                palette = img.getpalette()  # 获取原始调色板
                label_img = img.copy()     # 保持调色板模式
            else:
                label_img = img.convert('L')  # 非调色板图像转换为灰度
                palette = None

            width, height = label_img.size

            # 计算需要划分的行列数
            num_cols = math.ceil(width / 512)
            num_rows = math.ceil(height / 512)

            # 遍历所有行列位置
            for row in range(num_rows):
                for col in range(num_cols):
                    # 计算裁剪区域坐标
                    x_start = col * 512
                    y_start = row * 512
                    x_end = min(x_start + 512, width)
                    y_end = min(y_start + 512, height)

                    # 创建空白tile（根据图像模式）
                    if label_img.mode == 'P':
                        tile = Image.new('P', (512, 512), 0)
                        tile.putpalette(palette)  # 应用原始调色板
                    else:
                        tile = Image.new(label_img.mode, (512, 512), 0)

                    # 从原图裁剪实际区域
                    region = label_img.crop((x_start, y_start, x_end, y_end))
                    
                    # 将实际区域粘贴到空白tile
                    tile.paste(region, (0, 0))

                    # 生成新文件名
                    base_name = os.path.splitext(filename)[0]
                    new_filename = f"{base_name}_r{row+1:03d}_c{col+1:03d}.png"
                    output_path = os.path.join(output_dir, new_filename)

                    # 保存tile
                    tile.save(output_path, format='PNG')

if __name__ == "__main__":
    input_directory = r"data\\GID\\source_dir\\label_ori"
    output_directory = r"data\\GID\\source_dir\\label"
    
    # input_directory = r"data\\GID\\target_dir\\label_ori"
    # output_directory = r"data\\GID\\target_dir\\label"
    
    crop_and_save_label_tiles(input_directory, output_directory)