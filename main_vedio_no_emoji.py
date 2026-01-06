import cv2
import numpy as np
from keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
import os

# --- 配置区域 ---
MODEL_PATH = 'best_model_2.keras' 
EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

COLORS = {
    'Angry': (0, 0, 255),       # 红
    'Disgust': (0, 128, 0),     # 绿
    'Fear': (128, 0, 128),      # 紫
    'Happy': (0, 255, 255),     # 黄
    'Neutral': (200, 200, 200), # 灰
    'Sad': (255, 0, 0),         # 蓝
    'Surprise': (255, 165, 0),  # 橙
    'Uncertain': (100, 100, 100)# 深灰
}


# 置信度阈值，超过才判定
thresholds = {
    'Angry': 0.10, 'Disgust': 0.10, 'Fear': 0.15, 'Happy': 0.60, 
    'Neutral': 0.40, 'Sad': 0.20, 'Surprise': 0.50
}

if not os.path.exists(MODEL_PATH):
    print(f"❌ 错误: 找不到模型文件 {MODEL_PATH}")
    exit()

try:
    model = load_model(MODEL_PATH)
    print("✅ ResNet 模型加载成功！")
except Exception as e:
    print(f"❌ 模型加载失败: {e}")
    exit()

face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def draw_bar_chart(frame, x, y, w, h, preds):
    bar_x = x + w + 10       
    bar_w = 120              
    bar_h = 15               
    spacing = 5              
    
    if bar_x + bar_w > frame.shape[1]:
        bar_x = x - bar_w - 10

    for i, (label, prob) in enumerate(zip(EMOTION_LABELS, preds)):
        top = y + i * (bar_h + spacing)
        
        cv2.rectangle(frame, (bar_x, top), (bar_x + bar_w, top + bar_h), (40, 40, 40), -1)

        current_bar_width = int(bar_w * prob)
        color = COLORS.get(label, (255, 255, 255))
        
        if current_bar_width > 0:
            cv2.rectangle(frame, (bar_x, top), (bar_x + current_bar_width, top + bar_h), color, -1)
            
        text = f"{label[:3]}: {prob*100:.0f}%" 
        cv2.putText(frame, text, (bar_x, top + bar_h - 4), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: break
    
    frame = cv2.flip(frame, 1)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray_frame, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 1)


        roi_color = frame[y:y+h, x:x+w] 
        try:
            roi_rgb = cv2.cvtColor(roi_color, cv2.COLOR_BGR2RGB)
            roi_resized = cv2.resize(roi_rgb, (128, 128), interpolation=cv2.INTER_AREA)
            roi_array = np.expand_dims(roi_resized.astype('float32'), axis=0)
            roi_preprocessed = preprocess_input(roi_array)
            
  
            preds = model.predict(roi_preprocessed, verbose=0)[0]
            
            # --- 阈值判定逻辑 ---
            label_index = preds.argmax()   
            max_prob = preds[label_index]  
            raw_label_name = EMOTION_LABELS[label_index] 
            
            current_threshold = thresholds.get(raw_label_name, 0.3)
            
            if max_prob > current_threshold:
                # 确信：显示具体表情和颜色
                display_text = f"{raw_label_name} ({max_prob*100:.0f}%)"
                display_color = COLORS.get(raw_label_name)
            else:
                display_text = f"Uncertain ({max_prob*100:.0f}%)"
                display_color = COLORS['Uncertain']

            # 绘制头顶文字
            cv2.putText(frame, display_text, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, display_color, 2)
            
            # 绘制侧边条形图
            draw_bar_chart(frame, x, y, w, h, preds)
            
        except Exception as e:
            continue

    cv2.imshow('Emotion Analysis (Threshold Filtered)', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()