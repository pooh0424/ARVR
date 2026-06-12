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

def perspective_replacement(source_path, target_path, self_path, output_path):
    source_img = cv2.imread(source_path)  # 場景大圖
    target_img = cv2.imread(target_path)  # 要尋找的目標卡片
    self_img = cv2.imread(self_path)      # 準備替換上去的自己的照片

    if source_img is None or target_img is None or self_img is None:
        print("圖片讀取失敗，請檢查檔案路徑是否正確！")
        return

    # 將 Self Image 調整為與 Target Image 一模一樣的大小
    h, w = target_img.shape[:2]
    self_img_resized = cv2.resize(self_img, (w, h))

    # 轉為灰階影像以進行特徵點運算
    gray_source = cv2.cvtColor(source_img, cv2.COLOR_BGR2GRAY)
    gray_target = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)

    # Step 1: 找出特徵點與敘述子 (SIFT)
    sift = cv2.SIFT_create()
    kp_target, des_target = sift.detectAndCompute(gray_target, None)
    kp_source, des_source = sift.detectAndCompute(gray_source, None)

    # Step 2: 計算透視矩陣 (RANSAC + Homography)
    H, mask = find_homography(kp_target, des_target, kp_source, des_source, k=2, max_iters=5000, distance_threshold=5.0)

    if H is None:
        print("失敗：自定義 RANSAC 無法計算出有效的透視矩陣。")
        return

    # Step 3: 影像變換與貼合 (warpPerspective)
    height_src, width_src = source_img.shape[:2]
    warped_self = cv2.warpPerspective(self_img_resized, H, (width_src, height_src))
    mask_target = np.ones((h, w), dtype=np.uint8) * 255 # 建立一個與 Source Image 同樣大小的全白遮罩
    warped_mask = cv2.warpPerspective(mask_target, H, (width_src, height_src)) # 將遮罩也進行相同的透視變換
    
    # Step 4: 遮罩反轉，分離背景與前景
    inv_mask = cv2.bitwise_not(warped_mask)
    source_bg = cv2.bitwise_and(source_img, source_img, mask=inv_mask)
    self_fg = cv2.bitwise_and(warped_self, warped_self, mask=warped_mask)

    # Step 5: 融合
    result = cv2.add(source_bg, self_fg)
    cv2.imwrite(output_path, result)
    print("影像合成成功！檔案已儲存至:", output_path)

if __name__ == '__main__':
    SOURCE = 'test4/source.jpg'   # 場景圖
    TARGET = 'test4/target.jpg'   # 目標卡片圖
    SELF = 'test4/self.jpg'       # 要替換上去的照片
    OUTPUT = 'test4/output.jpg'   # 輸出的結果圖
    perspective_replacement(SOURCE, TARGET, SELF, OUTPUT)