import cv2
import numpy as np
import matplotlib.pyplot as plt

def compute_disparity_1d_ssd(img_left, img_right, window_size, max_disparity=32):
    h, w = img_left.shape
    disparity_map = np.zeros((h, w), dtype=np.float32) # Initialize the disparity map
    half_window = window_size // 2 # Calculate half of the window size for easier indexing
    img_left = img_left.astype(np.float32) # Convert images to float32 for accurate SSD calculation
    img_right = img_right.astype(np.float32) # Convert images to float32 for accurate SSD calculation

    for y in range(half_window, h - half_window): # Loop through each pixel in the left image, avoiding borders
        for x in range(half_window, w - half_window): 
            w_L = img_left[y-half_window : y+half_window+1, 
                           x-half_window : x+half_window+1] # Extract the window from the left image centered at (y, x)
            
            min_ssd = float('inf') # Initialize minimum SSD to infinity for comparison
            best_d = 0.0 # Initialize best disparity to zero
            
            for d in range(0,  max_disparity + 1): # Loop through possible disparities
                if x - d - half_window < 0: # Check if the shifted window in the right image goes out of bounds
                    continue # Skip this disparity if it goes out of bounds
                w_R = img_right[y-half_window : y+half_window+1, 
                                x-d-half_window : x-d+half_window+1] # Extract the corresponding window from the right image shifted by disparity d
                
                ssd = np.sum((w_L - w_R) ** 2) 
                
                if ssd < min_ssd:
                    min_ssd = ssd
                    best_d = d
                        
            disparity_map[y, x] = best_d # Assign the best disparity to the disparity map at (y, x)
    return disparity_map

if __name__ == "__main__":
    # Load the left and right images in grayscale
    img_l = cv2.imread('P_im0.ppm', cv2.IMREAD_GRAYSCALE)
    img_r = cv2.imread('P_im6.ppm', cv2.IMREAD_GRAYSCALE)
    if img_l is not None and img_r is not None:# Check if images are loaded successfully
        w_size = 11 # Window size for SSD, must be an odd number
        disparity_map = compute_disparity_1d_ssd(img_l, img_r, window_size=w_size) # Compute the disparity map
        plt.imshow(disparity_map, cmap='gray') # Display the disparity map
        plt.title(f'Disparity Map (Window Size = {w_size})')
        plt.colorbar()# Show colorbar for reference
        plt.show()