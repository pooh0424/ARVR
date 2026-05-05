import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Union

def process_and_resize_image(
    input_path: Union[str, Path], 
    output_path: Union[str, Path], 
    base_size: Tuple[int, int],
    target_size: Tuple[int, int],
    x_range: Tuple[int, int], 
    y_range: Tuple[int, int], 
    threshold: int
) -> bool:

    input_path = str(input_path)
    output_path = str(output_path)

    # 1. 讀取圖片
    img = cv2.imread(input_path)
    if img is None:
        print(f"⚠️ 無法讀取圖片: {input_path}，請檢查檔案是否存在。")
        return False

    # 2. 第一次強制縮放 (統一處理基準)
    img = cv2.resize(img, base_size)

    # 3. 降噪處理 (Gaussian Blur)
    img = cv2.GaussianBlur(img, (3, 3), 0)

    # 4. 轉為灰階
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 5. 定義範圍與邊界保護
    h, w = gray.shape
    x1, x2 = max(0, x_range[0]), min(w, x_range[1])
    y1, y2 = max(0, y_range[0]), min(h, y_range[1])

    # 準備輸出底板 (全黑)
    result = np.zeros_like(gray)

    # 提取感興趣區域 (ROI)
    roi = gray[y1:y2, x1:x2]
    
    # 在 ROI 內進行閾值過濾：大於 threshold 保留原值，否則設為 0
    roi_filtered = np.where(roi > threshold, roi, 0)
    
    # 將過濾後的 ROI 貼回黑色底板的對應位置
    result[y1:y2, x1:x2] = roi_filtered

    # 6. 第二次強制縮放 (Resize) 到最終目標大小
    final_result = cv2.resize(result, target_size)

    # 7. 儲存圖片
    cv2.imwrite(output_path, final_result)
    print(f"✅ 成功: {input_path} -> {output_path} (最終解析度: {final_result.shape[1]}x{final_result.shape[0]})")
    
    return True

def main():
    # --- 參數設定區 ---
    BASE_SIZE = (2483, 1147)    # 統一處理基準解析度 (寬, 高)
    TARGET_SIZE = (310, 143)    # 最終輸出的目標解析度 (寬, 高)
    X_LIM = (700, 1550)         # X 座標範圍 (左, 右)
    Y_LIM = (300, 880)          # Y 座標範圍 (上, 下)
    THRESH_VAL = 20             # 亮度閾值 (0-255)
    
    IMAGE_COUNT = 7             # 要處理的圖片數量 (im1.jpg ~ im7.jpg)

    print("--- 開始批次處理影像 ---")
    success_count = 0

    # --- 批次處理迴圈 ---
    for i in range(1, IMAGE_COUNT + 1):
        input_file = Path(f'im{i}.jpg')
        output_file = Path(f'pic{i}.bmp')
        
        is_success = process_and_resize_image(
            input_path=input_file, 
            output_path=output_file, 
            base_size=BASE_SIZE,
            target_size=TARGET_SIZE,
            x_range=X_LIM, 
            y_range=Y_LIM, 
            threshold=THRESH_VAL
        )
        
        if is_success:
            success_count += 1

    print(f"--- 處理完畢！共成功轉換 {success_count}/{IMAGE_COUNT} 張圖片 ---")

if __name__ == "__main__":
    main()