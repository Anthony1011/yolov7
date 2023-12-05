import os
import numpy as np

def get_file_list(input_path):
    #print(f"Read folder is : {input_path}")
    label_list = []
    for file in os.listdir(input_path):
        if file.endswith(".txt"):
            label_list.append(file)
    # print(label_list)
    return label_list


def transfer(input_path,output_path,mapping):
  
    txt_list = get_file_list(input_path)
    # print(txt_list)

    for txt_file in txt_list:

        with open(os.path.join(input_path,txt_file),'r') as f:
            lines = f.readlines()
        result = []

        for line in lines:
            values = line.strip().split()  # 移除換行符號並以空格分割
            result.append(values)

        if len(result)>0:
            for object in result:
                try:
                    class_id = int(object[0])
                    if class_id in mapping:
                        object[0]=str(mapping[class_id])
                        # if class_id == 6 or class_id == 7:
                        #     print(f"Class ID {class_id} in {txt_file}") 
                    else:
                        print(f"Class ID {class_id} in {txt_file} not found in mapping.")
                except ValueError as e: 
                    print(f"Error processing file {txt_file}: {e}")

        with open(os.path.join(output_path,txt_file), 'w') as new_labe:
            for line in result:
                # print(line)
                format_line = ' '.join(line)
                new_labe.write(format_line + '\n')
    
    print(f"The new labels were be storage in : {output_path}")
    print("Done")

if __name__ == "__main__":

    # Path setting
    input_path = "/media/itri/ADATA_SD700/KITTI/Lables/train_yolo"
    output_path = "/media/itri/ADATA_SD700/KITTI/Lables/train_yolo_and_list_after_coco" 
    
    # 更新的映射，包括所有KITTI类别到新索引的映射
    mapping = {
    0: 2,  # Car
    1: 80, # Van
    2: 7,  # Truck
    3: 0,  # Pedestrian
    4: 81, # Person_sitting
    5: 82, # Cyclist
    6: 83, # Tram
    7: 84  # Misc
    # 如果还有其他类别，也可以在这里添加
    }
    
    transfer(input_path,output_path,mapping)
