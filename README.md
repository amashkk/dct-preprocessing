# DCT 影像壓縮預處理方法

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

基於 Oizumi 論文 (IEEE 2006) 實作的 **DCT 影像壓縮選擇性預處理方法**，用於減少振鈴失真 (Ringing Artifacts)，並加入**自適應閾值優化**。

## 📋 專案概述

在 DCT 影像壓縮（如 JPEG）中，低位元率時常出現明顯的**振鈴失真**，尤其在強邊緣區域附近。直接使用全局低通濾波雖然可以減少失真，但會導致整體影像模糊。

本專案實作的**選擇性預處理方法**：
1. 計算每個像素的**修正自相關係數 (ρ_mod)**
2. **僅對邊緣區域**（容易產生振鈴的區域）進行低通濾波
3. 保留平滑區域和紋理不受影響

### 🚀 我們的優化：自適應閾值

我們在原始論文方法的基礎上，新增**自適應閾值機制**，根據影像特性（Y通道的全局變異數）動態調整濾波閾值，在感知品質 (SSIM) 和紋理保留上取得更好的效果。

## 🔬 方法論

### 修正自相關係數

演算法為每個像素計算 ρ_mod：

```
ρ_mod = R_xx(1) / (R_xx(0) + δ)
```

其中：
- `R_xx(0)` 和 `R_xx(1)` 是延遲為 0 和 1 的自相關值
- `δ` 是防止除以零的小常數

**解讀：**
- 較大的正 ρ_mod → 強邊緣，振鈴風險高 → 進行濾波
- 負的 ρ_mod → 高頻紋理區域 → 保留原始值

### 選擇性濾波流程

```
if ρ_mod > threshold:
    α = min(1.0, (ρ_mod - threshold) × filter_intensity)
    output = (1 - α) × original + α × filtered
else:
    output = original
```

### 自適應閾值（我們的優化）

根據影像變異數動態調整閾值，而非使用固定值：

```python
variance = np.var(Y_channel)
threshold = np.interp(variance, [500, 3000], [0.2, 0.5])
```

- **高變異數**（複雜紋理）→ 較高閾值 → 保護真實紋理
- **低變異數**（平滑區域）→ 較低閾值 → 更積極地降噪

## 📊 比較方法

| 方法 | 說明 |
|------|------|
| **直接壓縮** | 標準 DCT 壓縮，無預處理 |
| **固定閾值** | 原始論文方法 (ρ_threshold = 0.3) |
| **自適應閾值** | 我們的優化方法，動態調整閾值 |
| **全局濾波** | 對整張影像均勻進行低通濾波 |

## 🛠️ 安裝

### 相依套件

```bash
pip install numpy scipy pillow matplotlib scikit-image
```

### Google Colab 使用

直接開啟 notebook 並執行即可，所有相依套件已預先安裝。

## 📖 使用方式

### 快速開始 (Google Colab)

1. 在 Google Colab 開啟 `DCT_Preprocessing.ipynb`
2. 執行所有 cell
3. 依提示上傳影像
4. 查看四方比較結果

### Python 腳本

```python
from src.preprocessors import DCTPreprocessorAdaptive
from src.compressor import DCTCompressor

# 初始化
preprocessor = DCTPreprocessorAdaptive(
    base_rho_threshold=0.3,
    adaptive_threshold=True,
    adaptive_range_variance=(500, 3000),
    adaptive_range_threshold=(0.2, 0.5),
    filter_intensity=2.5,
    window_size=9
)
compressor = DCTCompressor(quality=10)

# 處理影像
preprocessed = preprocessor.preprocess_image(original_image)
compressed = compressor.compress_decompress(preprocessed)
```

## 📈 實驗結果

### 實驗設定
- **壓縮品質：** 10（極低，以便清楚呈現失真）
- **測試資料集：** [JPEG AI 測試影像](https://jpegai.github.io/test_images/)

### 範例結果

| 方法 | PSNR (dB) | SSIM | 邊緣保留 | 紋理保留 |
|------|-----------|------|----------|----------|
| 直接壓縮 | 30.23 | 0.9076 | 0.9537 | 0.9858 |
| 固定閾值 | 28.29 | 0.8882 | 0.8893 | 0.9688 |
| **自適應閾值** | **28.64** | **0.8926** | **0.9218** | **0.9791** |
| 全局濾波 | 27.69 | 0.8792 | 0.8617 | 0.9534 |

**主要發現：**
- 自適應閾值在 SSIM (+0.01) 和紋理保留 (+0.01) 上**優於**固定閾值
- 兩種選擇性方法相較直接壓縮都能**顯著減少振鈴失真**
- 全局濾波會導致**過度平滑**和細節損失

## 📁 專案結構

```
dct-preprocessing-project/
├── README.md                    # 專案說明（本文件）
├── requirements.txt             # Python 相依套件
├── DCT_Preprocessing.ipynb      # 主要 Colab notebook
├── src/
│   ├── __init__.py
│   ├── preprocessors.py         # 核心預處理類別
│   ├── compressor.py            # DCT 壓縮器實作
│   ├── metrics.py               # 品質指標 (PSNR, SSIM 等)
│   └── utils.py                 # 色彩空間轉換工具
├── examples/
│   └── demo.py                  # 使用範例
├── docs/
│   └── presentation.pdf         # 專案簡報
└── results/                     # 輸出結果
```

## 🔧 參數說明

### DCTPreprocessorAdaptive

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `base_rho_threshold` | 0.3 | ρ_mod 基礎閾值 |
| `adaptive_threshold` | True | 是否啟用自適應閾值 |
| `adaptive_range_variance` | (500, 3000) | 變異數插值範圍 |
| `adaptive_range_threshold` | (0.2, 0.5) | 閾值插值範圍 |
| `filter_intensity` | 2.5 | 濾波強度倍數 |
| `window_size` | 9 | 區域分析視窗大小 |
| `delta` | 10 | 正規化常數 |

### DCTCompressor

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `quality` | 50 | 類 JPEG 品質 (1-100) |

## 📚 參考文獻

[1] M. Oizumi, "Preprocessing method for DCT-based image-compression," *IEEE Transactions on Consumer Electronics*, vol. 52, no. 3, pp. 1021-1026, Aug. 2006, doi: [10.1109/TCE.2006.1706502](https://ieeexplore.ieee.org/document/1706502).

[2] Z. Wang, A. C. Bovik, H. R. Sheikh and E. P. Simoncelli, "Image quality assessment: from error visibility to structural similarity," *IEEE Transactions on Image Processing*, vol. 13, no. 4, pp. 600-612, April 2004.

[3] JPEG AI 測試影像資料集: https://jpegai.github.io/test_images/

