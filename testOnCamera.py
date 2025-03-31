import cv2
import torch
from torchvision import transforms, models
from PIL import Image
import numpy as np
from ultralytics import YOLO

emotion_labels = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Sad",
    5: "Surprise",
    6: "Neutral"
}

def get_convnext(model_size='large', num_classes=7):
    if model_size == 'tiny':
        model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights)
    elif model_size == 'small':
        model = models.convnext_small(weights=models.ConvNeXt_Small_Weights)
    elif model_size == 'base':
        model = models.convnext_base(weights=models.ConvNeXt_Base_Weights)
    else:
        model = models.convnext_large(weights=models.ConvNeXt_Large_Weights)
    model.classifier[2] = torch.nn.Linear(model.classifier[2].in_features, num_classes)
    return model

face_detector = YOLO('yolo_face_detection.pt')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
emotion_model = get_convnext(model_size='large', num_classes=7).to(device)
emotion_model.load_state_dict(torch.load(r"model_epoch_5.pth", map_location=device))  # 加载权重
emotion_model.eval()

# Preprocessing pipeline for ConvNeXt
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Open video capture (camera index 0: front camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Convert BGR (cv2) to RGB (for model inference)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Run face detection on the frame using YOLOv8
    results = face_detector.predict(source=rgb_frame, save=False, conf=0.5)  # 设置置信度阈值
    detections = results[0].boxes.xyxy.cpu().numpy()  # 获取检测框
    confidences = results[0].boxes.conf.cpu().numpy()  # 获取置信度
    classes = results[0].boxes.cls.cpu().numpy()  # 获取类别

    for det, conf, cls in zip(detections, confidences, classes):
        xmin, ymin, xmax, ymax = map(int, det)  # 转换为整数
        if conf < 0.5:  # 置信度过滤
            continue

        # Ensure coordinates are integer and within image bounds
        h, w, _ = frame.shape
        x1 = max(xmin, 0)
        y1 = max(ymin, 0)
        x2 = min(xmax, w - 1)
        y2 = min(ymax, h - 1)

        # Crop the face region from the original RGB frame.
        face_img = rgb_frame[y1:y2, x1:x2]
        if face_img.size == 0:
            continue
        
        # Convert cropped face to PIL Image
        face_pil = Image.fromarray(face_img)
        # Preprocess image: resize, normalize, etc.
        input_tensor = preprocess(face_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = emotion_model(input_tensor)
            prediction = int(logits.argmax(dim=1).item())

        label = emotion_labels.get(prediction, "Unknown")

        # Draw bounding box and label on the original frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow('Emotion Classification', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()