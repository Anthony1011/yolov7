import os


def gen_path(image_path,save_pth = "/home/itri/yolov7//train.txt"):
    # 打開一個文件來寫入
    with open(save_pth, 'w') as file:
        # 使用 os.walk 遍歷目錄
        for root, dirs, files in os.walk(image_path):
            for filename in files:
                # 構建文件的完整路徑
                file_path = os.path.join(root, filename)
                # 寫入到 save_ptht 文件
                file.write(file_path + '\n')    
            print(f"Save in {save_pth}")


if __name__ == "__main__":
    
    # 設置圖片路徑
    train ="train"
    val = "val"
    folder = val

    image_path = os.path.join("/home/itri/yolov7/dataset/LH/images",folder)
    save_pth = os.path.join("/home/itri/yolov7/dataset/LH",folder+".txt")

    gen_path(image_path,save_pth)

    # foler = ["train","val"]

    # for i in foler:
    #     gen_path(i)
    print(f"Done generate path")


