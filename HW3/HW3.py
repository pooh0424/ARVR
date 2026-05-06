import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

class PhotometricStereo:
    def __init__(self, data_dir):
        # 初始化 PhotometricStereo 類別，設定資料夾路徑並初始化相關屬性
        self.data_dir = data_dir 
        self.images = []
        self.light_dirs = []
        self.height = 0
        self.width = 0
        self.albedo = None
        self.normals = None
        self.height_map_naive = None
        self.height_map_poisson = None

    def load_data(self):
        print("--- 載入資料中 ---")
        light_path = os.path.join(self.data_dir, 'light.txt') #讀取光源方向的檔案路徑
        raw_lights = []

        with open(light_path, 'r') as f:
            for line in f:
                line = line.strip() #去除行首尾的空白字符
                if not line or ':' not in line: continue #跳過空行或不包含冒號的行
                content = line.split(':')[1].replace('(', '').replace(')', '').replace(',', ' ') #提取冒號後的內容並清理格式
                parts = content.split() #將清理後的內容分割成數字部分
                if len(parts) >= 3: 
                    raw_lights.append([float(x) for x in parts[:3]])

        temp_images = [] #暫存影像列表
        temp_lights = [] #暫存光源方向列表

        for i in range(1, len(raw_lights) + 1): #根據光源數量讀取對應的影像
            img_path = os.path.join(self.data_dir, f"pic{i}.bmp")
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) #以灰階模式讀取影像
            if img is not None:
                temp_images.append(img.astype(np.float32) / 255.0) #將影像轉換為浮點數並歸一化到 [0, 1]
                L = np.array(raw_lights[i-1], dtype=np.float32) #將光源方向轉換為 NumPy 陣列
                norm = np.linalg.norm(L) #計算光源方向的向量長度
                temp_lights.append(L / norm if norm != 0 else L) #將光源方向正規化（如果長度不為零）
            else:
                print(f"警告: 找不到 pic{i}.bmp")

        self.images = np.array(temp_images) #將暫存的影像列表轉換為 NumPy 陣列
        self.light_dirs = np.array(temp_lights) #將暫存的光源方向列表轉換為 NumPy 陣列
        self.height, self.width = self.images[0].shape #獲取影像的高度和寬度
        print(f"成功載入 {len(self.images)} 張影像。")

    def compute_ps(self):
        print("--- 計算法向量與 Albedo ---")
        I = self.images.reshape(len(self.images), -1) #將影像數據重塑為 (num_images, height*width) 的 2D 陣列
        S = self.light_dirs #光源方向矩陣

        S_inv = np.linalg.pinv(S) #計算光源方向矩陣的逆矩陣
        b = S_inv @ I #計算法向量矩陣 b

        albedo = np.linalg.norm(b, axis=0) #計算每個像素的 albedo（法向量的長度）
        self.albedo = albedo.reshape(self.height, self.width) #將 albedo 重塑回影像的形狀

        mask_flat = albedo > 1e-5 #創建一個遮罩，過濾掉 albedo 非常小的像素
        n = np.zeros_like(b) #初始化法向量矩陣
        n[2, :] = 1.0  #預設法向量的 z 分量為 1，避免除以零
        n[:, mask_flat] = b[:, mask_flat] / albedo[mask_flat] #對有效像素的法向量進行正規化

        self.normals = n.T.reshape(self.height, self.width, 3) #將法向量重塑回影像的形狀，並轉置以符合 (height, width, 3) 的格式

    # 方法一：傳統直接積分 (會產生斷層、條紋誤差)
    def integrate_naive(self, p, q): 
        h, w = p.shape #獲取 p 和 q 的高度和寬度
        Z = np.zeros((h, w)) #初始化高度圖 Z 為全零矩陣
        Z[0, :] = np.cumsum(p[0, :]) #從第一行開始，沿水平方向積分 p 梯度
        Z[1:, :] = Z[0, :] + np.cumsum(q[1:, :], axis=0) #從第二行開始，沿垂直方向積分 q 梯度，並加上第一行的積分結果
        return Z 

    # 方法二：Poisson 全域優化 (解決斷層問題)
    def integrate_poisson(self, p, q):
        h, w = p.shape #獲取 p 和 q 的高度和寬度
        dpdx = np.gradient(p, axis=1) #計算 p 梯度的 x 方向二階導數
        dqdy = np.gradient(q, axis=0) #計算 q 梯度的 y 方向二階導數
        f = dpdx + dqdy #計算 Poisson 方程的右側項 f

        F = np.fft.fft2(f) #對 f 進行 2D 快速傅立葉變換，轉換到頻域
        y, x = np.meshgrid(np.arange(h), np.arange(w), indexing='ij') #創建頻域的坐標網格
        
        denom = (2 * np.cos(2 * np.pi * x / w) - 2) + (2 * np.cos(2 * np.pi * y / h) - 2) #計算 Poisson 方程的頻域解的分母
        denom[0, 0] = 1  #避免除以零，將分母的 (0, 0) 項設置為 1 

        Z = np.real(np.fft.ifft2(F / denom)) #對頻域解進行逆傅立葉變換，獲得空間域的高度圖 Z
        return Z

    def reconstruct_surface(self):
        print("--- 執行兩種 3D 重建方法計算 ---")
        # 從法向量中提取 Nx、Ny、Nz 分量
        nx = self.normals[:, :, 0]
        ny = self.normals[:, :, 1]
        nz = self.normals[:, :, 2]

        nz_safe = np.where(nz == 0, 1e-5, nz) #避免除以零，將 Nz 中的零值替換為一個小的常數
        p = -nx / nz_safe   #計算 p 梯度（-Nx/Nz）
        q = ny / nz_safe    #計算 q 梯度（Ny/Nz）

        # 分別計算兩種高度圖
        Z_naive = self.integrate_naive(p, q)
        Z_poisson = self.integrate_poisson(p, q)

        
        mask = self.albedo > 0.1 #創建一個遮罩，過濾掉 albedo 小於等於 0.1 的像素，這些像素通常對重建結果貢獻較小且可能引入噪聲
        
        def process_Z(Z):
            valid_Z = Z[mask] #提取有效像素對應的高度值
            if len(valid_Z) > 0: #如果存在有效的高度值，則進行縮放處理
                z_min, z_max = np.percentile(valid_Z, 1), np.percentile(valid_Z, 99) #計算有效高度值的 1% 和 99% 百分位數，作為縮放的參考範圍
                Z = Z - z_min #將 Z 的最小值平移到 0
                z_range = z_max - z_min #計算有效高度值的範圍
                if z_range > 0: #如果範圍大於 0，則將 Z 縮放到合理的範圍內（這裡假設最大高度約為寬度的 20%）
                    Z = Z * ((self.width * 0.2) / z_range) #將 Z 縮放到寬度的 20% 範圍內
            Z[~mask] = 0 #將無效像素的高度值設置為 0
            return Z 

        self.height_map_naive = process_Z(Z_naive) #對 Naive 方法的高度圖進行處理，過濾無效像素並縮放到合理範圍
        self.height_map_poisson = process_Z(Z_poisson) #對 Poisson 方法的高度圖進行相同的處理

    def export_obj(self, Z_map, filename):
        print(f"--- 匯出模型至 {filename} ---") #根據提供的高度圖 Z_map 和輸出檔案名稱 filename，將重建的 3D 模型匯出為 OBJ 格式
        cx, cy = self.width / 2.0, self.height / 2.0 #計算圖像中心的坐標，將其作為 3D 模型的原點
        
        X, Y = np.meshgrid(np.arange(self.width), np.arange(self.height)) #創建 X 和 Y 的坐標網格，對應於圖像的寬度和高度
        X, Y = X - cx, (self.height - Y) - cy #將 X 和 Y 坐標平移，使圖像中心對齊到 (0, 0)，並將 Y 軸反轉以符合 3D 坐標系的慣例（Y 向上）
        Z = np.nan_to_num(Z_map, nan=0.0) #將 Z_map 中的 NaN 值替換為 0，確保高度圖中沒有無效值

        vertices = np.stack((X, Y, Z), axis=-1).reshape(-1, 3) #將 X、Y、Z 坐標堆疊成頂點列表，並重塑為 (num_vertices, 3) 的形狀
        r, c = np.arange(self.height - 1), np.arange(self.width - 1) #創建行和列的索引，用於定義面片的頂點索引
        R, C = np.meshgrid(r, c, indexing='ij') #創建行和列的索引網格，對應於圖像的高度和寬度，這些索引用於定義面片的頂點索引
        
        v1, v2 = R * self.width + C + 1, R * self.width + (C + 1) + 1 #計算面片的頂點索引，這裡使用 1-based 索引（OBJ 格式要求），v1 和 v2 分別對應於面片的兩個頂點
        v3, v4 = (R + 1) * self.width + C + 1, (R + 1) * self.width + (C + 1) + 1 #計算面片的另外兩個頂點索引，v3 和 v4 分別對應於面片的另外兩個頂點

        faces = np.vstack((np.stack((v1, v2, v3), axis=-1).reshape(-1, 3), 
                           np.stack((v2, v4, v3), axis=-1).reshape(-1, 3))) #
        #將兩種面片定義方式的頂點索引堆疊成一個面列表，這裡每個面由三個頂點索引組成，符合 OBJ 格式的要求
        with open(filename, 'w') as f:
            np.savetxt(f, vertices, fmt="v %.4f %.4f %.4f") #將頂點列表寫入 OBJ 文件，每行以 "v" 開頭，後面跟著頂點的 X、Y、Z 坐標，格式化為小數點後四位
            np.savetxt(f, faces, fmt="f %d %d %d") #將面列表寫入 OBJ 文件，每行以 "f" 開頭，後面跟著面片的三個頂點索引，格式化為整數

if __name__ == "__main__":
    folder = './test_datasets/bunny' #填入資料夾路徑
    folder_name = os.path.basename(os.path.normpath(folder)) #根據資料夾名稱動態命名輸出檔案

    ps = PhotometricStereo(folder) #實例化 PhotometricStereo 類別
    ps.load_data() #載入資料
    ps.compute_ps() #計算法向量與 Albedo
    ps.reconstruct_surface() #執行兩種 3D 重建方法計算
    
    # 根據資料夾名稱動態命名 3D 匯出檔案
    ps.export_obj(ps.height_map_naive, f"{folder_name}_naive.obj")
    ps.export_obj(ps.height_map_poisson, f"{folder_name}_poisson.obj")

    # 顯示 Albedo、Normal Map 以及法向量分量的組合圖
    fig = plt.figure(figsize=(15, 8))

    # 第一排：Albedo 與 Normal Map
    ax1 = fig.add_subplot(231)
    ax1.imshow(ps.albedo, cmap='gray')
    ax1.set_title('Albedo')
    ax1.axis('off')

    ax2 = fig.add_subplot(232)
    normal_rgb = (ps.normals + 1) / 2 
    ax2.imshow(normal_rgb)
    ax2.set_title('Normal Map (RGB)')
    ax2.axis('off')

    # 第二排：法向量分量 (Nx, Ny, Nz)
    ax3 = fig.add_subplot(234)
    im3 = ax3.imshow(ps.normals[:, :, 0], cmap='jet')
    ax3.set_title('Normal X (Nx)')
    fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

    ax4 = fig.add_subplot(235)
    im4 = ax4.imshow(ps.normals[:, :, 1], cmap='jet')
    ax4.set_title('Normal Y (Ny)')
    fig.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)

    ax5 = fig.add_subplot(236)
    im5 = ax5.imshow(ps.normals[:, :, 2], cmap='jet')
    ax5.set_title('Normal Z (Nz)')
    fig.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)

    plt.tight_layout() 
    
    # 根據資料夾名稱動態命名並儲存圖片
    save_img_path = f"{folder_name}_results.png"
    plt.savefig(save_img_path, dpi=300, bbox_inches='tight')
    print(f"--- 組合圖已成功儲存至: {save_img_path} ---")
    plt.show() #顯示圖像