import cv2
import numpy as np

def perspective_replacement_with_occlusion(source_path, target_path, self_path, output_path):
    # ==========================================
    # Step 0: 讀取圖片與前置處理
    # ==========================================
    source_img = cv2.imread(source_path)  # 場景大圖 (有手指遮擋)
    target_img = cv2.imread(target_path)  # 目標卡片 (乾淨的原始圖)
    self_img = cv2.imread(self_path)      # 準備替換上去的照片

    if source_img is None or target_img is None or self_img is None:
        print("圖片讀取失敗，請檢查檔案路徑是否正確！")
        return

    # 將自己的照片調整為與 Target Image 一模一樣的大小，確保轉換時四角對齊
    h, w = target_img.shape[:2]
    self_img_resized = cv2.resize(self_img, (w, h))

    # 轉為灰階影像以進行特徵點運算
    gray_source = cv2.cvtColor(source_img, cv2.COLOR_BGR2GRAY)
    gray_target = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)

    # ==========================================
    # Step 1 & 2: SIFT 特徵檢測與 KNN + RANSAC 匹配
    # ==========================================
    sift = cv2.SIFT_create()
    kp_target, des_target = sift.detectAndCompute(gray_target, None)
    kp_source, des_source = sift.detectAndCompute(gray_source, None)

    # 使用 BFMatcher 進行 KNN 匹配 (k=2)
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des_target, des_source, k=2)

    # Lowe's ratio test 過濾雜訊
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    print(f"找到 {len(good_matches)} 個良好的特徵匹配點。")

    if len(good_matches) >= 4:
        # 取得匹配點座標
        src_pts = np.float32([kp_target[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_source[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # ==========================================
        # Step 3: 計算透視矩陣 (Homography)
        # ==========================================
        # findHomography 自動利用 RANSAC 剔除錯誤對應點
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        # ==========================================
        # Step 4: 處理遮擋問題 (影像差異法 Background Subtraction)
        # ==========================================
        height_src, width_src = source_img.shape[:2]

        # 4-1. 建立基本的卡片遮罩，並投影到場景中
        mask_target_base = np.ones((h, w), dtype=np.uint8) * 255
        warped_card_mask = cv2.warpPerspective(mask_target_base, H, (width_src, height_src))

        # 4-2. 將乾淨的目標卡片投影到場景中，作為比對基準
        warped_clean_target = cv2.warpPerspective(target_img, H, (width_src, height_src))

        # 為了減少顏色造成的誤差，我們在灰階下進行比對
        gray_clean_target = cv2.cvtColor(warped_clean_target, cv2.COLOR_BGR2GRAY)
        
        # 只取卡片範圍內的影像進行比對
        source_roi = cv2.bitwise_and(gray_source, gray_source, mask=warped_card_mask)
        target_roi = cv2.bitwise_and(gray_clean_target, gray_clean_target, mask=warped_card_mask)

        # 4-3. 計算絕對差異 (差異大的地方就是手指等遮擋物)
        diff = cv2.absdiff(source_roi, target_roi)

        # 4-4. 二值化 (設定 Threshold，超過 50 的差異視為遮擋)
        # 注意：如果發現卡片原圖案被誤認為手指，請調高數值(如 70)；如果手指沒被挖乾淨，請調低(如 30)
        _, occlusion_mask = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)

        # 4-5. 形態學運算：消除雜訊並稍微擴張遮罩，讓手指邊緣更平滑
        kernel = np.ones((5, 5), np.uint8)
        occlusion_mask = cv2.morphologyEx(occlusion_mask, cv2.MORPH_OPEN, kernel)
        occlusion_mask = cv2.dilate(occlusion_mask, kernel, iterations=1)

        # ==========================================
        # Step 5: 產生最終遮罩並合成影像
        # ==========================================
        # 反轉遮擋遮罩：手指區域為 0 (黑)，無遮擋處為 255 (白)
        inv_occlusion_mask = cv2.bitwise_not(occlusion_mask)

        # 最終要貼上自拍照的範圍 = 「是卡片範圍」 且 「不是遮擋物」
        final_paste_mask = cv2.bitwise_and(warped_card_mask, inv_occlusion_mask)
        
        # 反轉最終遮罩，用來把大圖的目標位置「挖空」
        inv_final_paste_mask = cv2.bitwise_not(final_paste_mask)

        # 將自拍照變形到場景角度
        warped_self = cv2.warpPerspective(self_img_resized, H, (width_src, height_src))

        # 用遮罩合成影像
        source_bg = cv2.bitwise_and(source_img, source_img, mask=inv_final_paste_mask)
        self_fg = cv2.bitwise_and(warped_self, warped_self, mask=final_paste_mask)

        # 將挖空的背景與變形的自拍照相加
        result = cv2.add(source_bg, self_fg)

        # 儲存結果
        cv2.imwrite(output_path, result)
        print("影像合成成功！(含遮擋處理) 檔案已儲存至:", output_path)

        # 顯示圖片 (按任意鍵關閉視窗)
        # 設定預設顯示視窗大小 (例如 640x480)，避免在螢幕上顯示過大
        win_w, win_h = 640, 480

        # 1. 顯示差異圖 (Debug)
        cv2.namedWindow("1. Diff Map (Debug)", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("1. Diff Map (Debug)", win_w, win_h)
        cv2.imshow("1. Diff Map (Debug)", diff)

        # 2. 顯示抓出來的手指遮罩 (Debug)
        cv2.namedWindow("2. Occlusion Mask (Debug)", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("2. Occlusion Mask (Debug)", win_w, win_h)
        cv2.imshow("2. Occlusion Mask (Debug)", occlusion_mask)

        # 3. 顯示最終要貼上自拍照的範圍遮罩 (Debug)
        cv2.namedWindow("3. Final Paste Mask (Debug)", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("3. Final Paste Mask (Debug)", win_w, win_h)
        cv2.imshow("3. Final Paste Mask (Debug)", final_paste_mask)

        # 4. 顯示最終合成結果
        cv2.namedWindow("4. Final Result", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("4. Final Result", win_w, win_h)
        cv2.imshow("4. Final Result", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    else:
        print("失敗：找不到足夠的特徵點來進行轉換 (至少需要4點)。請確認圖片清晰度與特徵。")

# ==========================================
# 執行區域
# ==========================================
if __name__ == '__main__':
    # 請將以下檔名替換成你實際的圖片路徑
    SOURCE = 'self.jpg'   # 場景圖 (例如：內馬爾拿著書的那張)
    TARGET = 'target2.png'   # 目標卡片圖 (例如：書的正面原始圖片)
    SELF = 'target.jpg'       # 要替換上去的照片
    OUTPUT = 'output_occlusion.jpg'   # 輸出的結果圖

    perspective_replacement_with_occlusion(SOURCE, TARGET, SELF, OUTPUT)