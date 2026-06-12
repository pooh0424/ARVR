import cv2
import numpy as np
import itertools
import random

def find_homography(kp_target, des_target, kp_source, des_source, k=8, max_iters=50000, distance_threshold=5.0, overlap_threshold=0.03):
    n_target = len(kp_target)
    n_source = len(kp_source)
    
    if n_target < 4 or n_source < 4:
        print("[RANSAC] 特徵點數量不足，無法計算轉換矩陣。")
        return None, None
        
    bf = cv2.BFMatcher(cv2.NORM_L2)
    
    # 預先計算所有目標描述子在來源描述子中的最接近的描述子
    matches_1nn = bf.match(des_target, des_source)
    
    # 將所有特徵點對齊 queryIdx，便於在迴圈中利用 NumPy 進行高速矩陣運算
    matches_1nn_aligned = sorted(matches_1nn, key=lambda x: x.queryIdx)
    target_pts_all = np.float32([kp_target[m.queryIdx].pt for m in matches_1nn_aligned]).reshape(-1, 1, 2)
    source_pts_matched = np.float32([kp_source[m.trainIdx].pt for m in matches_1nn_aligned]) # 預先計算 source 描述子的平方範數，僅需計算一次
    source_sq = np.sum(des_source ** 2, axis=1) 
    
    # 預先品質篩選：依匹配距離排序，取得最優的前 300 個匹配點作為採樣池
    matches_1nn_sorted = sorted(matches_1nn, key=lambda x: x.distance)
    sample_pool = matches_1nn_sorted[:min(300, len(matches_1nn_sorted))]
    
    best_H = None
    best_ratio = 0.0
    
    # 進行 RANSAC 迭代
    for iteration in range(max_iters):
        # 隨機選取 4 個特徵點的匹配並提取其索引
        sampled_matches = random.sample(sample_pool, 4)
        indices = [m.queryIdx for m in sampled_matches]
        src_pts_4 = np.float32([kp_target[idx].pt for idx in indices])
        
        # 利用矩陣計算 4 個點與所有來源描述子的距離 (找kpd中最接近的前k個點)
        target_desc = des_target[indices] # shape (4, 128)
        target_sq = np.sum(target_desc ** 2, axis=1, keepdims=True) # shape (4, 1)
        dot_prod = np.dot(target_desc, des_source.T) # shape (4, N_source)
        dists_sq = target_sq + source_sq - 2 * dot_prod # shape (4, N_source)
        
        # 找出前 k 個最鄰近的來源特徵點索引
        top_k_idx = np.argsort(dists_sq, axis=1)[:, :k]
        
        c1 = [kp_source[i].pt for i in top_k_idx[0]]
        c2 = [kp_source[i].pt for i in top_k_idx[1]]
        c3 = [kp_source[i].pt for i in top_k_idx[2]]
        c4 = [kp_source[i].pt for i in top_k_idx[3]]
            
        # 確保每個挑選的點都有足夠的候選點
        if len(c1) < k or len(c2) < k or len(c3) < k or len(c4) < k:
            continue
            
        # 測試 k^4 種組合
        for p1, p2, p3, p4 in itertools.product(c1, c2, c3, c4):
            dst_pts_4 = np.float32([p1, p2, p3, p4])
            try:
                # 計算單應性矩陣
                H = cv2.getPerspectiveTransform(src_pts_4, dst_pts_4)
            except cv2.error:
                continue

            if H is None:
                continue

            # 將轉換套用到全部的 target_pts
            warped_pts = cv2.perspectiveTransform(target_pts_all, H).reshape(-1, 2)
            # 計算轉換後的點與原本匹配點之間的距離
            dists = np.linalg.norm(warped_pts - source_pts_matched, axis=1)
            
            # 統計在門檻值內的點數量比例 (即重疊比例)
            inliers = dists < distance_threshold
            ratio = np.sum(inliers) / len(target_pts_all)
            
            # 尋找重疊比例最高的轉換
            if ratio > best_ratio:
                best_ratio = ratio
                best_H = H
                
        # 若重疊比例已高於一個極佳的預期值，即可提早結束迭代
        if best_ratio >= 0.12:
            break
            
    print(f"[RANSAC] 迭代次數: {iteration + 1}, 最佳重疊比例: {best_ratio:.4f}")
    
    if best_ratio >= overlap_threshold:
        return best_H, None
    else:
        print(f"[RANSAC] 警告: 最佳重疊比例 {best_ratio:.4f} 未達門檻 {overlap_threshold}。")
        return best_H, None


# ==========================================
# 參數獨立區塊 (Configuration)
# ==========================================
PARAMS = {
    "paths": {
        "source": 'test5/source.jpg',    # 場景大圖
        "target": 'test5/target.png',  # 參考卡片圖
        "self_photo": 'test5/self.jpg',     # 要貼上去的照片
        "output": 'test5/output_optimized.jpg'
    },
    "sift": {
        "ransac_reproj": 5.0     # RANSAC 門檻
    },
    "occlusion": {
        "blur_size": 5,         # 高斯模糊大小 (需為奇數)
        "morph_kernel": 5,       # 形態學運算核大小
        "dilate_iter": 3,        # 膨脹次數
        "initial_threshold": 70  # 初始差異門檻值 
    }
}



def perspective_replacement_with_interactive_threshold():
    # 讀取圖片
    source_img = cv2.imread(PARAMS["paths"]["source"])
    target_img = cv2.imread(PARAMS["paths"]["target"])
    self_img = cv2.imread(PARAMS["paths"]["self_photo"])

    if source_img is None or target_img is None or self_img is None:
        print("圖片讀取失敗，請檢查檔案路徑！")
        return

    h, w = target_img.shape[:2]
    self_img_resized = cv2.resize(self_img, (w, h))
    gray_source = cv2.cvtColor(source_img, cv2.COLOR_BGR2GRAY)
    gray_target = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)

    # Step 1: 使用 SIFT 演算法偵測特徵點並計算描述子
    sift = cv2.SIFT_create()
    kp_target, des_target = sift.detectAndCompute(gray_target, None)
    kp_source, des_source = sift.detectAndCompute(gray_source, None)

    # Step 2: 使用 RANSAC 演算法估計單應性矩陣 H
    H, _ = find_homography(kp_target, des_target, kp_source, des_source, k=2, max_iters=5000, distance_threshold=PARAMS["sift"]["ransac_reproj"])
    if H is None:
        print("失敗：自定義 RANSAC 無法計算出有效的透視矩陣。")
        return

    # Step 3: 預處理遮擋計算與影像貼合所需之遮罩與參考影像
    height_src, width_src = source_img.shape[:2]
    mask_target_base = np.ones((h, w), dtype=np.uint8) * 255
    warped_card_mask = cv2.warpPerspective(mask_target_base, H, (width_src, height_src))
    warped_clean_target = cv2.warpPerspective(target_img, H, (width_src, height_src))
    
    # Step 4: 將調整好大小的自己照片透視投影至場景空間中 (warped_self)
    warped_self = cv2.warpPerspective(self_img_resized, H, (width_src, height_src))

    # Step 5: 預先進行高斯模糊並計算 HSV 空間的絕對差值 (以偵測手指等遮擋物)
    k_size = (PARAMS["occlusion"]["blur_size"], PARAMS["occlusion"]["blur_size"])
    blur_source = cv2.GaussianBlur(source_img, k_size, 0)
    blur_target = cv2.GaussianBlur(warped_clean_target, k_size, 0)
    hsv_source = cv2.cvtColor(blur_source, cv2.COLOR_BGR2HSV)
    hsv_target = cv2.cvtColor(blur_target, cv2.COLOR_BGR2HSV)
    source_roi = cv2.bitwise_and(hsv_source, hsv_source, mask=warped_card_mask)
    target_roi = cv2.bitwise_and(hsv_target, hsv_target, mask=warped_card_mask)
    diff_hsv = cv2.absdiff(source_roi, target_roi)
    
    # 提取 H 與 S 通道的差值
    diff_h = diff_hsv[:, :, 0].astype(np.float32)
    diff_s = diff_hsv[:, :, 1]
    
    # 1. 修正 Hue 通道的環狀邊界差值 (最大差值為 90)
    diff_h = np.minimum(diff_h, 180.0 - diff_h)
    # 將 Hue 差值比例拉伸至 0-255 區間
    diff_h_scaled = np.clip(diff_h * 2.8, 0, 255).astype(np.uint8)
    
    # 2. 低飽和度過濾：若場景圖或參考卡片圖的飽和度低於 30，則將 Hue 差值歸零，避免無彩色區的色彩噪點干擾
    s_source = hsv_source[:, :, 1]
    s_target = hsv_target[:, :, 1]
    low_sat_mask = (s_source < 30) | (s_target < 30)
    diff_h_scaled[low_sat_mask] = 0
    
    # 3. 進行加權融合：30% Hue 差值 + 70% Saturation 差值，降低不穩定 H 通道的權重並忽略 V (亮度)
    diff = cv2.addWeighted(diff_h_scaled, 0.3, diff_s, 0.7, 0)

    # Step 6: 讀取設定參數並進行遮擋遮罩計算
    thresh_val = PARAMS["occlusion"]["initial_threshold"]

    # 1. 藉由二值化閾值處理 (cv2.threshold) 偵測出高於門檻的遮擋物區域。
    # 2. 進行閉運算 (MORPH_CLOSE) 填補遮擋物內部的微小破洞與空隙。
    # 3. 進行開運算 (MORPH_OPEN) 去除背景中的細微雜訊噪點。
    # 4. 進行影像膨脹 (cv2.dilate)，向外擴張以保證手指邊緣能夠被完美包含。
    _, occlusion_mask = cv2.threshold(diff, thresh_val, 255, cv2.THRESH_BINARY)
    kernel = np.ones((PARAMS["occlusion"]["morph_kernel"], PARAMS["occlusion"]["morph_kernel"]), np.uint8)
    occlusion_mask = cv2.morphologyEx(occlusion_mask, cv2.MORPH_CLOSE, kernel)
    occlusion_mask = cv2.morphologyEx(occlusion_mask, cv2.MORPH_OPEN, kernel)
    occlusion_mask = cv2.dilate(occlusion_mask, kernel, iterations=PARAMS["occlusion"]["dilate_iter"])

    # Step 7: 影像合成 (使用 bitwise 運算進行硬邊融合)
    final_mask = cv2.bitwise_and(warped_card_mask, cv2.bitwise_not(occlusion_mask))
    inv_mask = cv2.bitwise_not(final_mask)
    source_bg = cv2.bitwise_and(source_img, source_img, mask=inv_mask)
    self_fg = cv2.bitwise_and(warped_self, warped_self, mask=final_mask)
    result = cv2.add(source_bg, self_fg)

    # Step 8: 儲存合成結果至指定路徑
    cv2.imwrite(PARAMS["paths"]["output"], result)
    print(f"成功儲存結果至 {PARAMS['paths']['output']} (Threshold: {thresh_val})")


if __name__ == '__main__':
    perspective_replacement_with_interactive_threshold()