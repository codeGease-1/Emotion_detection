import cv2
import numpy as np
from keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
import os

# --- 配置区域 ---
MODEL_PATH = 'best_model_2.keras' 
ASSETS_DIR = 'assets'           

EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

emoji_animations = {}

print(f"正在从以下目录加载资源: {os.path.abspath(ASSETS_DIR)}") 

for label in EMOTION_LABELS:
    folder_path = os.path.join(ASSETS_DIR, label.lower()) 
    frames = []
    
    if os.path.exists(folder_path):
        valid_extensions = ('.png', '.jpg', '.jpeg')
        file_list = [f for f in os.listdir(folder_path) if f.lower().endswith(valid_extensions)]
        file_list = sorted(file_list) # 排序，保证动画顺序
        
        for filename in file_list:
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            if img is not None:
                frames.append(img)
        
        if len(frames) > 0:
            emoji_animations[label] = frames
            print(f"✅ {label}: 成功加载 {len(frames)} 帧")
        else:
            print(f"⚠️ {label}: 文件夹存在但未找到图片 (检查是否为png/jpg)")
            print(f"   -> 正在扫描路径: {os.path.abspath(folder_path)}")
    else:
        print(f"❌ {label}: 未找到文件夹 {folder_path}")

# 加载模型
try:
    model = load_model(MODEL_PATH)
    print("ResNet 模型加载成功！")
except Exception as e:
    print(f"模型加载失败: {e}")
    exit()

face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

# 定义差异化阈值
thresholds = {
    'Angry': 0, 'Disgust': 0, 'Fear': 0.15,
    'Happy': 0.85, 'Neutral': 0.65, 'Sad': 0.01, 'Surprise': 0.65
}

def overlay_transparent(background, overlay, x, y, w, h):
    """ 图片叠加函数 (带透明通道 + 边界保护) """
    try:
        if overlay.shape[2] == 3:
            overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2BGRA)

        overlay_resized = cv2.resize(overlay, (w, h), interpolation=cv2.INTER_AREA)
        
        b, g, r, a = cv2.split(overlay_resized)
        alpha_mask = a / 255.0
        beta_mask = 1.0 - alpha_mask
        
        y1, y2 = max(0, y), min(background.shape[0], y + h)
        x1, x2 = max(0, x), min(background.shape[1], x + w)
        
        if y1 >= y2 or x1 >= x2: return background

        oy1, oy2 = y1 - y, y2 - y
        ox1, ox2 = x1 - x, x2 - x
        
        roi_bg = background[y1:y2, x1:x2]
        roi_overlay = overlay_resized[oy1:oy2, ox1:ox2]
        mask_alpha = alpha_mask[oy1:oy2, ox1:ox2]
        mask_beta = beta_mask[oy1:oy2, ox1:ox2]

        for c in range(3):
            roi_bg[:, :, c] = (roi_overlay[:, :, c] * mask_alpha + roi_bg[:, :, c] * mask_beta)
            
        background[y1:y2, x1:x2] = roi_bg
    except Exception as e:
        pass 
    return background

frame_counter = 0

while True:
    ret, frame = cap.read()
    if not ret: break
    
    frame = cv2.flip(frame, 1) # 镜像翻转，体验更好
    frame_counter += 1

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray_frame, 1.3, 5)

    for (x, y, w, h) in faces:
        # --- 预测逻辑 ---
        # 1. 截取彩色图
        roi_color = frame[y:y+h, x:x+w] 
        # 2. 转 RGB 
        roi_rgb = cv2.cvtColor(roi_color, cv2.COLOR_BGR2RGB)
        # 3. 缩放到 128x128
        roi_resized = cv2.resize(roi_rgb, (128, 128), interpolation=cv2.INTER_AREA)
        # 4. 预处理
        roi_array = np.expand_dims(roi_resized.astype('float32'), axis=0)
        roi_preprocessed = preprocess_input(roi_array)
        
        preds = model.predict(roi_preprocessed, verbose=0)[0]
        label_index = preds.argmax()
        confidence = preds[label_index]
        label_name = EMOTION_LABELS[label_index]

        # --- 动画播放逻辑 ---
        current_threshold = thresholds.get(label_name, 0.4)
        
        # 只有当 (置信度够高) 且 (该表情有动画素材) 时才显示动画
        if confidence > current_threshold and label_name in emoji_animations:
            animation_frames = emoji_animations[label_name]
            total_frames = len(animation_frames)
            
            if total_frames > 0:
                speed_factor = 2 
                current_anim_index = (frame_counter // speed_factor) % total_frames
                emoji_overlay = animation_frames[current_anim_index]

                scale = 1.4 
                new_w, new_h = int(w * scale), int(h * scale)
                new_x = x - int((new_w - w) / 2)
                new_y = y - int((new_h - h) / 2)

                frame = overlay_transparent(frame, emoji_overlay, new_x, new_y, new_w, new_h)
        
        else:
            # 没素材或不确信时，画框显示文字
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, f"{label_name}: {confidence*100:.0f}%", (x, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv2.imshow('Animated Emoji Overlay', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()