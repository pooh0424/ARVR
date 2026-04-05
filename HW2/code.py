import cv2
import numpy as np
import matplotlib.pyplot as plt

def compute_disparity_1d_ssd(img_left, img_right, window_size, max_disparity=64):
    h, w = img_left.shape
    disparity_map = np.zeros((h, w), dtype=np.float32)
    half_window = window_size // 2
    
    img_left = img_left.astype(np.float32)
    img_right = img_right.astype(np.float32)

    # 逐像素掃描
    for y in range(half_window, h - half_window):
        for x in range(half_window, w - half_window):
            
            # 取得左圖的 Window
            w_L = img_left[y-half_window : y+half_window+1, 
                           x-half_window : x+half_window+1]
            
            min_ssd = float('inf')
            best_d = 0
            
            # 限制只在「同一條水平線」上，往左邊搜尋視差 d
            for d in range(0, x - half_window + 1):
                if x - d - half_window >= 0:
                    # 取得右圖的 Window
                    w_R = img_right[y-half_window : y+half_window+1, 
                                    x-d-half_window : x-d+half_window+1]
                    
                    # 計算 SSD Error [cite: 13, 14]
                    ssd = np.sum((w_L - w_R) ** 2)
                    
                    # 尋找 SSD 最小值的視差
                    if ssd < min_ssd:
                        min_ssd = ssd
                        best_d = d
                        
            disparity_map[y, x] = best_d
            
    return disparity_map

# 讀取圖片 (請確保檔名與路徑正確)
img_l = cv2.imread('im0.ppm', cv2.IMREAD_GRAYSCALE)
img_r = cv2.imread('im8.ppm', cv2.IMREAD_GRAYSCALE)

if img_l is not None and img_r is not None:
    w_size = 17
    depth_map = compute_disparity_1d_ssd(img_l, img_r, window_size=w_size)

    plt.imshow(depth_map, cmap='gray')
    plt.title(f'Disparity Map (Window Size = {w_size})')
    plt.colorbar()
    plt.show()