# 面部表情识别与动态 Emoji 互动系统 (Emotion-Emoji-Sync)

本项目是一个基于深度学习的实时面部情绪识别系统。通过摄像头捕捉人脸，利用训练好的深度学习模型识别表情，并实时在画面中叠加相应的动态 Emoji 覆盖。

## 🛠️ 技术实现
* **核心模型**：基于 **ResNet50** 架构的迁移学习（Transfer Learning）。
* **后端框架**：TensorFlow / Keras。
* **视觉处理**：使用 **MediaPipe/OpenCV** 进行高频人脸关键点检测与图像预处理。
* **数据策略**：针对 RAF-DB 数据集执行**离线数据增强**，并应用 `class_weight` 解决样本不平衡问题。

## 📂 文件夹结构
* `/assets`: 存放各情绪对应的 PNG 透明图序列
* `main.py`: 本地实时运行脚本，包含摄像头调用与 Emoji 渲染逻辑。
* `trainmodel.ipynb`: 记录了从数据加载、模型构建、微调到验证的完整训练过程。
* `requirements.txt`: 项目运行所需的 Python 依赖库。

## 🚀 快速开始
1. **克隆项目**: `git clone https://github.com/codeGease-1/Emotion_detection.git`
2. **安装依赖**: `pip install -r requirements.txt`
3. **下载模型**: 这里提供一个训练好的模型，可下载 `best_model_2.keras` 并放入项目根目录。
4. **运行程序**: `python main.py`

## ⚠️ 开发反思
* **识别精度**: 受限于 RAF-DB 数据集的多样性，模型在“恐惧”与“惊讶”表情上存在一定的语义重叠，导致识别结果偶尔跳变。

## 📜 许可证
本项目遵循 [MIT License](LICENSE) 协议。