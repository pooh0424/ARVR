import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

class PhotometricStereo:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.images = []       
        self.light_dirs = []   
        self.height = 0
        self.width = 0
        self.albedo = None     
        self.normals = None    
        self.height_map = None 

    def load_data(self):
        """1. 讀取光源檔案與影像"""
        print(f"--- 載入資料中 ---")
        light_path = os.path.join(self.data_dir, 'light.txt')
        
        # 解析 light.txt
        raw_lights = []
        with open(light_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or ':' not in line: continue
                # 取得冒號後的內容並移除括號與逗號
                content = line.split(':')[1].replace('(', '').replace(')', '').replace(',', ' ')
                parts = content.split()
                if len(parts) >= 3:
                    raw_lights.append([float(x) for x in parts[:3]])
        # 載入影像並同步過濾光源
        temp_images = []
        temp_lights = []
        for i in range(1, len(raw_lights) + 1):
            img_name = f"pic{i}.bmp"
            img_path = os.path.join(self.data_dir, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            if img is not None:
                temp_images.append(img.astype(np.float32) / 255.0)
                # 歸一化光源向量
                L = np.array(raw_lights[i-1], dtype=np.float32)
                norm = np.linalg.norm(L)
                temp_lights.append(L / norm if norm != 0 else L)
            else:
                print(f"警告: 找不到 {img_name}")
        self.images = np.array(temp_images)
        self.light_dirs = np.array(temp_lights)
        self.height, self.width = self.images[0].shape
        print(f"成功載入 {len(self.images)} 張影像。")
        
    def compute_ps(self):
        """2. 計算 Albedo 與 Normals (加入安全遮罩)"""
        print("--- 計算表面法向量 ---")
        I = self.images.reshape(len(self.images), -1)
        S = self.light_dirs
        
        S_inv = np.linalg.pinv(S)
        b = S_inv @ I 
        
        # 1. 計算反射率
        albedo = np.linalg.norm(b, axis=0)
        self.albedo = albedo.reshape(self.height, self.width)
        
        # 2. 建立有效像素遮罩 (門檻值可依資料微調)
        mask_flat = albedo > 0.1
        
        # 3. 初始化法向量，預設為 [0, 0, 1] 而非 [0, 0, 0]
        n_raw = np.zeros_like(b)
        n_raw[2, :] = 1.0  # 預設 Z 軸方向為 1
        
        # 4. 只在有效區域計算正規化
        n_raw[:, mask_flat] = b[:, mask_flat] / albedo[mask_flat]
        
        self.normals = n_raw.T.reshape(self.height, self.width, 3)

    def reconstruct_surface(self):
        """3. 使用 Mask 排除背景的積分"""
        print("--- 重建表面高度 (已套用 Mask) ---")
        nx = self.normals[:, :, 0]
        ny = self.normals[:, :, 1]
        nz = self.normals[:, :, 2]
        
        # 防止除以 0，且只在物體區域計算斜率
        nz_safe = np.where(nz == 0, 1.0, nz)
        p = -nx / nz_safe
        q = -ny / nz_safe
        
        # 建立 2D 遮罩
        mask = self.albedo > 0.1
        
        Z = np.zeros((self.height, self.width))
        
        # --- 改良版積分路徑 ---
        # 1. 處理第一欄 (Column 0)
        for r in range(1, self.height):
            if mask[r, 0]:
                Z[r, 0] = Z[r-1, 0] + q[r, 0]
            else:
                Z[r, 0] = Z[r-1, 0] # 背景維持高度
        
        # 2. 處理每一列 (從左向右)
        for r in range(self.height):
            for c in range(1, self.width):
                if mask[r, c]:
                    Z[r, c] = Z[r, c-1] + p[r, c]
                else:
                    Z[r, c] = Z[r, c-1] # 背景維持高度
        
        # 3. (選配) 背景歸零，讓物體在 OBJ 中更明顯
        Z[~mask] = 0
        self.height_map = Z

    def export_obj(self, filename="result.obj"):
        """4. 輸出包含 v 與 f 的 OBJ 檔"""
        print(f"--- 匯出 OBJ 至 {filename} ---")
        with open(filename, 'w') as f:
            # 寫入頂點 v
            for r in range(self.height):
                for c in range(self.width):
                    f.write(f"v {c} {self.height - r} {self.height_map[r, c]}\n")
            
            # 寫入面 f (巢狀迴圈連接像素)
            for r in range(self.height - 1):
                for c in range(self.width - 1):
                    # OBJ 索引從 1 開始
                    v1 = (r * self.width) + c + 1
                    v2 = (r * self.width) + (c + 1) + 1
                    v3 = ((r + 1) * self.width) + c + 1
                    v4 = ((r + 1) * self.width) + (c + 1) + 1
                    # 兩個三角形組成一個正方形像素格
                    f.write(f"f {v1} {v2} {v3}\n")
                    f.write(f"f {v2} {v4} {v3}\n")
        print("匯出完成。")

if __name__ == "__main__":
    # 請確保此路徑下有 light.txt 和 pic1.bmp ~ pic6.bmp
    folder = './test_datasets/bunny/' 
    
    ps = PhotometricStereo(folder)
    ps.load_data()
    ps.compute_ps()
    ps.reconstruct_surface()
    ps.export_obj("output_model.obj")
    
    # 顯示預覽圖
    plt.subplot(121), plt.imshow(ps.albedo, cmap='gray'), plt.title('Albedo')
    plt.subplot(122), plt.imshow((ps.normals + 1) / 2), plt.title('Normal Map')
    plt.show()