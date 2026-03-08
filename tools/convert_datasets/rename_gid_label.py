import os
from pathlib import Path

def auto_rename_labels(folder_path):
    """
    自动移除文件名中的'_5label'部分（无确认提示）
    :param folder_path: 目标文件夹路径 (e.g. 'data\GID\source_dir\label_merged')
    """
    target_folder = Path(folder_path)
    
    if not target_folder.exists():
        raise FileNotFoundError(f"目录不存在: {target_folder}")
    
    rename_count = 0
    skip_count = 0

    for file_path in target_folder.glob('*.*'):
        if '_24label' in file_path.name:
            new_name = file_path.name.replace('_24label', '', 1)  # 只替换第一个匹配项
            new_path = file_path.with_name(new_name)
            
            if not new_path.exists():
                file_path.rename(new_path)
                rename_count += 1
            else:
                skip_count += 1

    print(f"操作完成: 成功重命名 {rename_count} 个文件，跳过 {skip_count} 个已存在文件")

# 直接执行
if __name__ == "__main__":
    target_dir = r"data\GID\source_dir\label"
    # target_dir = r"data\GID\target_dir\label"
    try:
        auto_rename_labels(target_dir)
    except Exception as e:
        print(f"错误发生: {str(e)}")