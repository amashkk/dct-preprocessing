"""
示範腳本：展示如何使用 DCT 預處理模組。

本腳本示範四種方法的比較：
1. 直接壓縮（無預處理）
2. 固定閾值預處理（原始論文方法）
3. 自適應閾值預處理（我們的優化）
4. 全局濾波預處理（基準比較）

使用方式：
    python demo.py --image 影像路徑.jpg --quality 10
"""

import argparse
import sys
import os

# 將父目錄加入路徑以便匯入
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from src.preprocessors import DCTPreprocessorFixed, DCTPreprocessorAdaptive, UniformPreprocessor
from src.compressor import DCTCompressor
from src.metrics import calculate_psnr, calculate_ssim, calculate_edge_preservation, calculate_texture_preservation, calculate_sharpness


def run_comparison(image_path, quality=10, output_path=None):
    """
    對影像執行四方比較。
    
    參數
    ----------
    image_path : str
        輸入影像路徑。
    quality : int
        DCT 壓縮品質 (1-100)。
    output_path : str, optional
        比較圖儲存路徑。
    """
    # 載入影像
    print(f"載入影像: {image_path}")
    img = Image.open(image_path)
    original = np.array(img)
    
    # 處理 RGBA
    if len(original.shape) == 3 and original.shape[2] == 4:
        original = original[:, :, :3]
        print("  (已將 RGBA 轉換為 RGB)")
    
    print(f"影像尺寸: {original.shape}")
    
    # 初始化元件
    print(f"\n使用壓縮品質: {quality}")
    compressor = DCTCompressor(quality=quality)
    
    fixed_preprocessor = DCTPreprocessorFixed(
        rho_threshold=0.3,
        filter_intensity=2.5,
        window_size=9,
        delta=10
    )
    
    adaptive_preprocessor = DCTPreprocessorAdaptive(
        base_rho_threshold=0.3,
        adaptive_threshold=True,
        adaptive_range_variance=(500, 3000),
        adaptive_range_threshold=(0.2, 0.5),
        filter_intensity=2.5,
        window_size=9,
        delta=10
    )
    
    uniform_preprocessor = UniformPreprocessor(filter_strength=1.0)
    
    # 方法 1：直接壓縮
    print("\n[1/4] 直接壓縮（無預處理）...")
    compressed = compressor.compress_decompress(original.astype(np.float64))
    psnr_direct = calculate_psnr(original, compressed)
    ssim_direct = calculate_ssim(original, compressed)
    print(f"  PSNR: {psnr_direct:.2f} dB | SSIM: {ssim_direct:.4f}")
    
    # 方法 2：固定閾值
    print("\n[2/4] 固定閾值預處理...")
    preprocessed_fixed = fixed_preprocessor.preprocess_image(original.astype(np.float64))
    compressed_fixed = compressor.compress_decompress(preprocessed_fixed)
    psnr_fixed = calculate_psnr(original, compressed_fixed)
    ssim_fixed = calculate_ssim(original, compressed_fixed)
    filtered_percent_fixed = (fixed_preprocessor.filter_map > 0.1).sum() / fixed_preprocessor.filter_map.size * 100
    print(f"  PSNR: {psnr_fixed:.2f} dB | SSIM: {ssim_fixed:.4f} | 濾波區域: {filtered_percent_fixed:.1f}%")
    
    # 方法 3：自適應閾值
    print("\n[3/4] 自適應閾值預處理...")
    preprocessed_adaptive = adaptive_preprocessor.preprocess_image(original.astype(np.float64), verbose=True)
    compressed_adaptive = compressor.compress_decompress(preprocessed_adaptive)
    psnr_adaptive = calculate_psnr(original, compressed_adaptive)
    ssim_adaptive = calculate_ssim(original, compressed_adaptive)
    filtered_percent_adaptive = (adaptive_preprocessor.filter_map > 0.1).sum() / adaptive_preprocessor.filter_map.size * 100
    print(f"  PSNR: {psnr_adaptive:.2f} dB | SSIM: {ssim_adaptive:.4f} | 濾波區域: {filtered_percent_adaptive:.1f}%")
    
    # 方法 4：全局濾波
    print("\n[4/4] 全局濾波預處理...")
    preprocessed_uniform = uniform_preprocessor.preprocess_image(original.astype(np.float64))
    compressed_uniform = compressor.compress_decompress(preprocessed_uniform)
    psnr_uniform = calculate_psnr(original, compressed_uniform)
    ssim_uniform = calculate_ssim(original, compressed_uniform)
    print(f"  PSNR: {psnr_uniform:.2f} dB | SSIM: {ssim_uniform:.4f}")
    
    # 結果摘要
    print("\n" + "="*60)
    print("結果摘要")
    print("="*60)
    print(f"\n{'方法':<20} {'PSNR (dB)':<12} {'SSIM':<10}")
    print("-"*50)
    print(f"{'1. 直接壓縮':<20} {psnr_direct:<12.2f} {ssim_direct:<10.4f}")
    print(f"{'2. 固定閾值':<20} {psnr_fixed:<12.2f} {ssim_fixed:<10.4f}")
    print(f"{'3. 自適應閾值':<20} {psnr_adaptive:<12.2f} {ssim_adaptive:<10.4f}")
    print(f"{'4. 全局濾波':<20} {psnr_uniform:<12.2f} {ssim_uniform:<10.4f}")
    
    # 分析
    print("\n" + "-"*60)
    print("優化分析：自適應 vs 固定")
    print("-"*60)
    print(f"  PSNR 增益: {psnr_adaptive - psnr_fixed:+.3f} dB")
    print(f"  SSIM 增益: {ssim_adaptive - ssim_fixed:+.5f}")
    print(f"  動態閾值: {adaptive_preprocessor.current_rho_threshold:.4f} (vs 固定 0.3000)")
    
    if ssim_adaptive > ssim_fixed:
        print("\n  ✓ 自適應閾值達到更好的感知品質 (SSIM)！")
    
    # 建立視覺化
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    axes[0, 0].imshow(original.astype(np.uint8))
    axes[0, 0].set_title('原始影像')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(compressed.astype(np.uint8))
    axes[0, 1].set_title(f'1. 直接壓縮\nPSNR: {psnr_direct:.2f} dB')
    axes[0, 1].axis('off')
    
    axes[1, 0].imshow(compressed_fixed.astype(np.uint8))
    axes[1, 0].set_title(f'2. 固定閾值\nPSNR: {psnr_fixed:.2f} dB')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(compressed_adaptive.astype(np.uint8))
    axes[1, 1].set_title(f'3. 自適應閾值\nPSNR: {psnr_adaptive:.2f} dB')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n比較圖已儲存至: {output_path}")
    else:
        plt.show()
    
    return {
        'direct': {'psnr': psnr_direct, 'ssim': ssim_direct},
        'fixed': {'psnr': psnr_fixed, 'ssim': ssim_fixed},
        'adaptive': {'psnr': psnr_adaptive, 'ssim': ssim_adaptive},
        'uniform': {'psnr': psnr_uniform, 'ssim': ssim_uniform},
    }


def main():
    parser = argparse.ArgumentParser(
        description='DCT 預處理示範 - 比較不同預處理方法'
    )
    parser.add_argument('--image', '-i', required=True, help='輸入影像路徑')
    parser.add_argument('--quality', '-q', type=int, default=10, 
                        help='壓縮品質 (1-100，預設: 10)')
    parser.add_argument('--output', '-o', help='比較圖儲存路徑')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.image):
        print(f"錯誤：找不到影像檔案: {args.image}")
        sys.exit(1)
    
    run_comparison(args.image, args.quality, args.output)


if __name__ == '__main__':
    main()
