import cv2
import torch
from models.yolo import Model  # 根據 YOLOv7 的具體實現來調整
import numpy as np

# 確保 CUDA 可用，如果不可用則使用 CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device,torch.cuda.is_available(),torch.cuda.get_device_name(0),torch.__version__)

# 加載模型
checkpoint = torch.load('./weights/yolov7.pt', map_location=device)

# 判斷加載的對象是模型還是狀態字典
if isinstance(checkpoint, dict) and 'model' in checkpoint:
    # 如果是字典且包含 'model' 鍵，則這是完整的模型
    model = checkpoint['model']
elif isinstance(checkpoint, dict):
    # 如果是狀態字典
    model = Model('./cfg/training/yolov7.yaml', nc=80)  # 創建模型實例
    model.load_state_dict(checkpoint)
else:
    # 如果加載的是完整模型
    model = checkpoint

# 將模型轉移到指定設備
model.to(device).eval()

# 開啟攝像頭
cap = cv2.VideoCapture(0)  # 0 通常是內置攝像頭

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # frame = cv2.imread("./inference/images/image3.jpg")
    # frame = cv2.resize(frame, (640, 640))  # 調整影像大小
    
    # 將 BGR 影像轉換為 RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 將影像從 HWC 轉換為 CHW 格式
    img = np.transpose(frame_rgb, (2, 0, 1))  # 將通道轉換到前面
    img = torch.from_numpy(img).to(device).float()
    img /= 255.0  # 正規化
    img = img.unsqueeze(0)  # 增加批次維度

    # 轉換為半精度 (half precision)
    img = img.half()

    # 模型推論
    with torch.no_grad():
        pred = model(img)[0]
        # print(pred)

        # 假设模型一共可以识别的类别数量
        num_classes = 80
        names = [ 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
         'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
         'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
         'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
         'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
         'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
         'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
         'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
         'hair drier', 'toothbrush' ]
        
        # 遍历每个检测到的物体
        for i in range(pred.size(1)):  # pred.size(1) 是检测到的物体的数量
            # 提取第 i 个物体的检测信息
            detection = pred[0, i]  # 这里的 '0' 是批次的索引

        # for i, detection in enumerate(pred):
            # 提取边界框坐标（假设坐标已经被缩放到了原始图像尺寸）
            x1, y1, x2, y2 = detection[0:4].tolist()  # 将坐标转换为 Python 列表
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)

            # 提取置信度
            confidence = detection[4].item()  # 将置信度转换为 Python 数值

            # 提取类别概率
            class_probabilities = detection[5:num_classes+5].tolist()  # 将类别概率转换为 Python 列表

            # 找到概率最高的类别索引和概率值
            best_class_index = int(np.argmax(class_probabilities))  # 使用 np.argmax 找到最高概率的索引
            conf = class_probabilities[best_class_index]  # 获取最高概率值
            cls_name = names[best_class_index]

            # 打印检测到的物体的信息
            print(f"Object {i}:")
            print(f"  Coordinates: ({x1}, {y1}), ({x2}, {y2})")
            print(f"  Best Class : {cls_name}")
            print(f"  Best class probability: {conf}")

            #Plot image
            print(frame.shape)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # cv2.putText(img, str(conf), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        frame = cv2.flip(frame,1)
        cv2.imshow('YOLOv7 Person Detection', frame)
        # 按 'q' 鍵退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# 釋放攝像頭資源並關閉窗口
cap.release()
cv2.destroyAllWindows()
