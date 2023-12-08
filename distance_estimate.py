import cv2
import torch
import random
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.torch_utils import select_device
# from utils.plots import plot_one_box

def get_dis(x):
    f=2.86 # cm
    D = (f * 5.2 )/(int(x[3])-int(x[1])) 
    return D

def plot_one_box(x, d, img, color=None, label=None, line_thickness=3):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
        cv2.putText(img, str(d), (c1[0], c1[1] - 15), 0, tl / 3, [225, 0, 0], thickness=tf, lineType=cv2.LINE_AA)

def detect():
    weights = './weights/yolov7.pt'  # 模型权重文件
    img_size = 640
    conf_thres = 0.25
    iou_thres = 0.45
    # device = 'cpu'  # 或 'cpu'
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(device)


    # 初始化
    device = select_device(device)
    model = attempt_load(weights, map_location=device)  # 加载模型
    model.to(device).eval()  # 设置为评估模式
    stride = int(model.stride.max())  # 模型步幅

    # 打开摄像头
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 图像预处理
        img = cv2.resize(frame, (img_size, img_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img).to(device).float() / 255.0  # 归一化
        img = img.permute(2, 0, 1).unsqueeze(0)  # 调整通道顺序并增加批次维度
        
        classes = [39]
        # 推理
        with torch.no_grad():
            pred = model(img)[0]
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes,agnostic=False)
            

        # 绘制边界框 
        for i, det in enumerate(pred):  # 每张图片的检测结果
            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()  # 调整坐标到原始图片尺寸
                for *xyxy, conf, cls in reversed(det):
                    label = f'{model.names[int(cls)]} {conf:.2f}'
                    x1, y1, x2, y2 = xyxy
                    d = get_dis(xyxy)
                    plot_one_box(xyxy, d, frame, label=label, color=(0, 255, 0), line_thickness=1)
                    print(f"Object detected: \n Coordinates: ({x1}, {y1}), ({x2}, {y2})\n Confidence: {conf}\n Class: {cls}")


        cv2.imshow('YOLOv7 Detection', frame)

        # 按 'q' 键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()
if __name__ == "__main__":

    # 执行检测
    print(torch.cuda.is_available())
    print(torch.cuda.get_device_name(0))
    print(torch.__version__)
    detect()
