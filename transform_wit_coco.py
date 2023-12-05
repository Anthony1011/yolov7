import os

# 更新的映射，包括所有KITTI类别到新索引的映射
mapping = {
    0: 2,  # Car
    1: 81, # Van
    2: 7,  # Truck
    3: 0,  # Pedestrian
    4: 82, # Person_sitting
    5: 83, # Cyclist
    6: 84, # Tram
    7: 85  # Misc
    # 如果还有其他类别，也可以在这里添加
}

# 標籤文件所在的目錄
label_dir = "/media/itri/ADATA_SD700/KITTI/Lables/labels/train_yolo"  # 請替換為您的標籤目錄路徑
new_label_dir = "/media/itri/ADATA_SD700/KITTI/Lables/labels/train_yolo_and_list_after_coco"  # 輸出新標籤的目錄

#  確保新標籤目錄存在
os.makedirs(new_label_dir, exist_ok=True)

# 遍歷標籤文件
for label_file in os.listdir(label_dir):
    if label_file.endswith(".txt"):
        with open(os.path.join(label_dir, label_file), 'r') as file:
            lines = file.readlines()

        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) > 0:
                class_id = int(parts[0])
                if class_id in mapping:
                    # 更新類別索引
                    parts[0] = str(mapping[class_id])
                new_line = ' '.join(parts)
                new_lines.append(new_line)

        # 寫入新標籤文件
        with open(os.path.join(new_label_dir, label_file), 'w') as file:
            file.write('\n'.join(new_lines))