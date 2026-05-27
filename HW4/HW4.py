import cv2
import numpy as np

def perspective_replacement(source_path, target_path, self_path, output_path):
    # ==========================================
    # 讀取圖片 (Input)
    # ==========================================
    source_img = cv2.imread(source_path)  # 場景大圖
    target_img = cv2.imread(target_path)  # 要尋找的目標卡片
    self_img = cv2.imread(self_path)      # 準備替換上去的自己的照片

    if source_img is None or target_img is None or self_img is None:
        print("圖片讀取失敗，請檢查檔案路徑是否正確！")
        return

    # 【重點提醒】將 Self Image 調整為與 Target Image 一模一樣的大小
    # 這樣後續套用同一個透視矩陣 (Homography) 時，四個角才能完美對齊
    h, w = target_img.shape[:2]
    self_img_resized = cv2.resize(self_img, (w, h))

    # 轉為灰階影像以進行特徵點運算
    gray_source = cv2.cvtColor(source_img, cv2.COLOR_BGR2GRAY)
    gray_target = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)

    # ==========================================
    # Step 1: 找出特徵點與敘述子 (SIFT)
    # ==========================================
    sift = cv2.SIFT_create()
    kp_target, des_target = sift.detectAndCompute(gray_target, None)
    kp_source, des_source = sift.detectAndCompute(gray_source, None)

    # ==========================================
    # Step 2: 特徵匹配與過濾 (KNN + Lowe's Ratio Test)
    # ==========================================
    # 使用 BFMatcher 進行 KNN 匹配 (k=2 表示找出最近的兩個點)
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des_target, des_source, k=2)

    # 使用 Lowe's ratio test 過濾掉雜訊匹配點
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    print(f"找到 {len(good_matches)} 個良好的特徵匹配點。")

    # 確保匹配點數量足夠計算透視矩陣 (至少需要 4 個點)
    if len(good_matches) >= 4:
        # 取得匹配點在圖片中的 X, Y 座標
        src_pts = np.float32([kp_target[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_source[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # ==========================================
        # Step 3: 計算透視矩陣 (RANSAC + Homography)
        # ==========================================
        # findHomography 內建 RANSAC 演算法，會自動剔除錯誤對應，計算出最佳的 3x3 轉換矩陣 H
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        # ==========================================
        # Step 4: 影像變換與貼合 (warpPerspective)
        # ==========================================
        height_src, width_src = source_img.shape[:2]
        
        # 1. 將你的照片 (已 resize 過) 投影到 Source 影像的三維空間中
        warped_self = cv2.warpPerspective(self_img_resized, H, (width_src, height_src))

        # 2. 製作遮罩 (Mask)：為了把 Source 圖片裡原本的卡片「挖空」
        # 產生一張與 Target 大小相同，但全白的圖片 (數值為 255)
        mask_target = np.ones((h, w), dtype=np.uint8) * 255
        # 將遮罩一樣投影到 Source 空間
        warped_mask = cv2.warpPerspective(mask_target, H, (width_src, height_src))

        # 反轉遮罩：卡片區域變黑(0)，背景變白(255)
        inv_mask = cv2.bitwise_not(warped_mask)

        # 3. 影像合成
        # 用反轉遮罩去挖空 Source 圖片 (卡片處變黑)
        source_bg = cv2.bitwise_and(source_img, source_img, mask=inv_mask)
        # 用正遮罩確保自己的圖片周圍是乾淨的黑色
        self_fg = cv2.bitwise_and(warped_self, warped_self, mask=warped_mask)

        # 將挖空的背景與變形後的自己的照片相加
        result = cv2.add(source_bg, self_fg)

        # 儲存並顯示結果
        cv2.imwrite(output_path, result)
        print("影像合成成功！檔案已儲存至:", output_path)

        # 顯示圖片 (按下任一鍵關閉視窗)
        # 設定預設顯示視窗大小 (例如 640x480)，避免在螢幕上顯示過大
        win_w, win_h = 640, 480
        cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Result", win_w, win_h)
        cv2.imshow("Result", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    else:
        print("失敗：找不到足夠的特徵點來進行轉換 (至少需要4點)。請嘗試解析度更高、特徵更明顯的圖片。")


# ==========================================
# 執行區域
# ==========================================
if __name__ == '__main__':
    # 請將以下檔名替換成你實際的圖片路徑
    SOURCE = 'source.jpg'   # 場景圖
    TARGET = 'target.jpg'   # 目標卡片圖
    SELF = 'self.jpg'       # 要替換上去的照片
    OUTPUT = 'output.jpg'   # 輸出的結果圖

    perspective_replacement(SOURCE, TARGET, SELF, OUTPUT)