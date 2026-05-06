# AOI-defection-classification

# Deep Learning for Computer Vision Guide

這是一份專為 Python 初學者設計的電腦視覺（CV）與深度學習基礎指南。內容涵蓋了從數據處理到模型優化的核心管道，特別適用於自動光學檢測 (AOI) 等工業瑕疵辨識應用。

---

## 目錄
1. [基本的矩陣概念：圖片在電腦中是什麼？](#1-基本的矩陣概念圖片在電腦中是什麼)
2. [Python 進階語法：Pandas 與 os/path](#2-python-進階語法pandas-與-ospath)
3. [卷積神經網絡 (CNN) 原理](#3-卷積神經網絡-cnn-原理)
4. [遷移學習 (Transfer Learning) 與 Fine-tuning](#4-遷移學習-transfer-learning-與-fine-tuning)
5. [過擬合 (Overfitting) 的處理策略](#5-過擬合-overfitting-處理策略)

---

## 1. 基本的矩陣概念：圖片在電腦中是什麼？

**使用場景：** 讀取影像、調整解析度、以及理解模型輸入的維度規範。

**原理邏輯：** 圖片在電腦中是以 Tensor (張量) 形式存在的數值矩陣。
* **維度結構：** (Height, Width, Channels)。
* **像素值：** 通常為 0（全黑）到 255（全白）的整數。
* **RGB 通道：** 彩色圖片包含紅、綠、藍三層矩陣疊加。
* **關鍵參數：** target_size (模型要求的輸入尺寸), interpolation (縮放時的插值演算法)。

```python
import cv2
import numpy as np

# 讀取圖片並查看其矩陣形狀
img = cv2.imread('sample.jpg')
print(f"原始形狀: {img.shape}") # (高, 寬, 通道)

# 為了符合模型輸入 (如 DenseNet)，將圖片縮放至 224x224
resized_img = cv2.resize(img, (224, 224))
print(f"縮放後形狀: {resized_img.shape}")
```
## 2. Python 進階語法：Pandas 與 os/path

**使用場景：** 當你的影像資料集很大時，用表格管理標籤 (Label) 與對應的檔案路徑。

**原理邏輯：**
* **os.path：** 用於自動合成路徑，避免在 Windows (使用 \) 與 Linux (使用 /) 之間切換時發生路徑格式錯誤。
* **Pandas：** 像高效的電子表格軟體一樣處理數據，進行資料過濾 (Filtering) 與標籤映射 (Mapping)。

```python
import os
import pandas as pd

# 安全地合成路徑，不用擔心作業系統的斜線方向
img_dir = "./data/images"
df = pd.read_csv("train.csv")

# 數據過濾：只篩選出 Label 為 0 (正常品) 的資料
normal_df = df[df['label'] == 0]

# 動態生成完整路徑欄位
df['path'] = df['id'].apply(lambda x: os.path.join(img_dir, f"{x}.png"))
```

## 3. 卷積神經網絡 (CNN) 原理

**使用場景：** 自動化特徵提取，辨識影像中的瑕疵（例如邊緣、形狀、特定紋理）。

**核心組件：**
* **Conv2D (卷積層)：** 使用多組濾波器 (Filters) 掃描影像，捕捉特徵點。
* **MaxPooling2D (池化層)：** 壓縮特徵圖並保留最顯著資訊，減少運算量並防止特徵位置偏移造成的誤差。
* **Dense (全連接層)：** 位於模型末端，負責最終的決策邏輯，判斷特徵組合屬於哪種分類。

```python
from tensorflow.keras import layers, models

model = models.Sequential([
    # 卷積層：提取基礎特徵
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    # 池化層：濃縮資訊、減少計算量
    layers.MaxPooling2D((2, 2)),
    
    # 攤平層：將二維特徵圖轉換為一維向量
    layers.Flatten(),
    
    # 全連接層：根據特徵進行判斷
    layers.Dense(64, activation='relu'),
    # 輸出層：6 個節點代表 6 種瑕疵類別的機率
    layers.Dense(6, activation='softmax')
])
```

## 4. 遷移學習 (Transfer Learning) 與 Fine-tuning

**使用場景：** 當你的訓練資料不足（例如瑕疵樣本極少，只有幾十或幾百張）時，借用大廠訓練好的模型記憶。

**原理邏輯：**
* **預訓練模型：** 使用在 ImageNet（包含千萬張圖片）上訓練過的模型（如 DenseNet169），它已經具備辨識線條、形狀與基礎物件的能力。
* **凍結層 (Freezing)：** 在訓練初期鎖定預訓練層的權重不變，只針對你的瑕疵類型訓練最後的分類輸出層。
* **微調 (Fine-tuning)：** 在模型初步穩定後，解凍部分層級，並以「極小」的學習率微調模型，使其更精確地適應特定產業的影像。

```python
from tensorflow.keras.applications import DenseNet169
from tensorflow.keras.models import Model
from tensorflow.keras import layers

# 1. 載入預訓練模型，不包含原有的 1000 類分類頭
base_model = DenseNet169(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# 2. 第一階段：凍結底層權重
base_model.trainable = False

# 3. 接上自定義的分類頭
x = layers.GlobalAveragePooling2D()(base_model.output)
outputs = layers.Dense(6, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=outputs)

# 註：第二階段 (Fine-tuning) 通常在訓練幾輪後才進行
# base_model.trainable = True
# for layer in base_model.layers[:-20]: # 僅解凍最後 20 層
#     layer.trainable = False
```

## 5. 過擬合 (Overfitting) 的處理策略

**使用場景：** 當模型在訓練集表現完美（死記硬背），但在沒看過的測試集表現極差時。

**解決工具：**
* **Data Augmentation (資料擴增)：** 透過隨機旋轉、翻轉、平移，強迫模型學習特徵的「本質」而非特定位置。
* **Dropout：** 在訓練過程中隨機「關掉」部分神經元，增加模型的強健性，防止對特定路徑的過度依賴。
* **EarlyStopping：** 監控驗證集的分數，若連續多輪不再進步，則提早終止訓練，防止模型開始死背噪音。

| 方法 | 說明 |
| :--- | :--- |
| 資料擴增 | 增加圖片樣式多樣性，減少模型死背固定角度 |
| Dropout | 訓練時隨機斷開神經元連結，增加模型穩定性 |
| EarlyStopping | 在驗證集表現開始變差前及時收手停止訓練 |

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping

# 1. 資料增強：讓考題變豐富，防止模型死背
train_datagen = ImageDataGenerator(
    rotation_range=20,      # 隨機旋轉 20 度
    width_shift_range=0.1,  # 隨機水平平移
    horizontal_flip=True    # 隨機水平翻轉
)

# 2. 提早停止：監控驗證集損失 (val_loss)
# 如果連續 5 輪 (patience) 沒進步就停止訓練
early_stop = EarlyStopping(
    monitor='val_loss', 
    patience=5, 
    restore_best_weights=True
)
```
