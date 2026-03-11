#!/usr/bin/env python
import torch
import torch.nn as nn
import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import random
import time
from tqdm import tqdm

# --- CONFIGURATION ---
# UPDATED PATHS (Spring_26)
TRAIN_DIR = "/home/adipat/Documents/Spring_26/CV/P1/Deep_homography/Deep-Homography/Phase2/Data/Train_Phase1/Set1"
VAL_DIR   = "/home/adipat/Documents/Spring_26/CV/P1/Deep_homography/Deep-Homography/Phase2/Data/Val"
# Using Val as Test since no separate Ground Truth Test set exists
TEST_DIR  = "/home/adipat/Documents/Spring_26/CV/P1/Deep_homography/Deep-Homography/Phase2/Data/Val" 

# UNSUPERVISED MODEL PATH
MODEL_PATH = "/home/adipat/Documents/Spring_26/CV/P1/Deep_homography/Deep-Homography/model_weights/unsupervised_homography_net.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 1. MODEL ARCHITECTURE
# ==========================================
class HomographyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(2,64,3,padding=1,bias=False), nn.BatchNorm2d(64), nn.ReLU(True), nn.Conv2d(64,64,3,padding=1,bias=False), nn.BatchNorm2d(64), nn.ReLU(True), nn.MaxPool2d(2,2))
        self.layer2 = nn.Sequential(nn.Conv2d(64,64,3,padding=1,bias=False), nn.BatchNorm2d(64), nn.ReLU(True), nn.Conv2d(64,64,3,padding=1,bias=False), nn.BatchNorm2d(64), nn.ReLU(True), nn.MaxPool2d(2,2))
        self.layer3 = nn.Sequential(nn.Conv2d(64,128,3,padding=1,bias=False), nn.BatchNorm2d(128), nn.ReLU(True), nn.Conv2d(128,128,3,padding=1,bias=False), nn.BatchNorm2d(128), nn.ReLU(True), nn.MaxPool2d(2,2))
        self.layer4 = nn.Sequential(nn.Conv2d(128,128,3,padding=1,bias=False), nn.BatchNorm2d(128), nn.ReLU(True), nn.Conv2d(128,128,3,padding=1,bias=False), nn.BatchNorm2d(128), nn.ReLU(True))
        
        # Include Dropout to match saved weights, but .eval() will disable it during inference
        self.fc = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(128*16*16, 1024), 
            nn.ReLU(True), 
            nn.Dropout(0.1),
            nn.Linear(1024, 8)
        )

    def forward(self, x):
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        return self.fc(x.contiguous().view(x.size(0), -1))

class UnifiedHomographyModel(nn.Module):
    def __init__(self, model_type="Sup"):
        super().__init__()
        self.regressor = HomographyNet()
        self.model_type = model_type
        
    def forward(self, stacked, rho=32):
        pred = self.regressor(stacked)
        if self.model_type == "Sup": 
            return pred.view(-1, 4, 2) * float(rho)
        else: 
            # UNSUPERVISED INFERENCE: Use Tanh + Scale
            return (torch.tanh(pred) * float(rho)).view(-1, 4, 2)

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def get_classical_homography(imgA, imgB):
    """ Calculates Homography using ORB + RANSAC (Phase 1 Baseline) """
    orb = cv2.ORB_create(nfeatures=2000)
    kp1, des1 = orb.detectAndCompute(imgA, None)
    kp2, des2 = orb.detectAndCompute(imgB, None)
    
    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4: 
        return np.zeros((4, 2))
    
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)[:100]
    
    if len(matches) < 4: 
        return np.zeros((4, 2))

    src = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    
    H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    
    if H is None: 
        return np.zeros((4, 2))

    # Convert 3x3 H matrix to 4-point Delta for comparison
    base = np.array([[0,0], [128,0], [128,128], [0,128]], dtype=np.float32).reshape(-1, 1, 2)
    est = cv2.perspectiveTransform(base, H)
    return est.reshape(4, 2) - base.reshape(4, 2)

def get_sample(path, crop_size=128, rho=32):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None: return None
    H, W = img.shape
    if H < 200 or W < 200: return None
    
    x = np.random.randint(rho, W - crop_size - rho)
    y = np.random.randint(rho, H - crop_size - rho)
    
    patch_a = img[y:y+crop_size, x:x+crop_size]
    corners_a = np.array([[x,y], [x+crop_size-1,y], [x+crop_size-1,y+crop_size-1], [x,y+crop_size-1]], dtype=np.float32)
    
    gt_delta = np.random.randint(-rho+1, rho, (4, 2)).astype(np.float32)
    corners_b = corners_a + gt_delta
    
    H_mat = cv2.getPerspectiveTransform(corners_a, corners_b)
    H_inv = np.linalg.inv(H_mat)
    warped = cv2.warpPerspective(img, H_inv, (W,H), flags=cv2.WARP_INVERSE_MAP)
    patch_b = warped[y:y+crop_size, x:x+crop_size]
    
    return patch_a, patch_b, gt_delta

def plot_overlay(ax, delta, color, label, linestyle='-'):
    base = np.array([[0,0], [128,0], [128,128], [0,128]], dtype=np.float32)
    poly = base + delta
    poly = np.vstack([poly, poly[0]])
    ax.plot(poly[:,0], poly[:,1], color=color, linewidth=2, linestyle=linestyle, label=label)

def run_epe_benchmark(name, model, folder_path, num_samples=200):
    files = glob.glob(os.path.join(folder_path, "*"))
    if len(files) == 0:
        print(f"[WARN] No files in {folder_path}. Skipping.")
        return

    errors = []
    times = []
    
    model.eval()
    with torch.no_grad():
        for _ in range(num_samples):
            path = random.choice(files)
            sample = get_sample(path)
            if sample is None: continue
            pa, pb, gt = sample
            
            # Prepare Input
            stacked = np.stack([pa.astype(np.float32)/255., pb.astype(np.float32)/255.], axis=0)
            tensor = torch.from_numpy(stacked).unsqueeze(0).float().to(device)
            
            # Run Inference & Time it
            start = time.time()
            pred = model(tensor, rho=32).cpu().numpy().squeeze()
            if torch.cuda.is_available(): torch.cuda.synchronize()
            end = time.time()
            
            # Calculate EPE (End Point Error)
            diff = pred - gt
            epe = np.mean(np.linalg.norm(diff, axis=1))
            
            errors.append(epe)
            times.append((end - start) * 1000)

    if errors:
        print(f"Set: {name:10} | Avg EPE: {np.mean(errors):.2f} px | Avg Time: {np.mean(times):.2f} ms")

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model file not found: {MODEL_PATH}")
        return

    print("--- 1. LOADING UNSUPERVISED MODEL ---")
    # Initialize with "Unsup" to ensure correct forward pass (Tanh)
    unsup_model = UnifiedHomographyModel("Unsup").to(device)
    
    state = torch.load(MODEL_PATH, map_location=device)
    try: unsup_model.regressor.load_state_dict(state)
    except: unsup_model.load_state_dict(state, strict=False)
    
    unsup_model.eval() # Important! Disables dropout
    print(f"Loaded Unsupervised from {MODEL_PATH}")

    print("\n--- 2. NETWORK ARCHITECTURE ---")
    print(unsup_model.regressor)

    print("\n--- 3. GENERATING VISUALIZATIONS (Fig 18 - Unsupervised) ---")
    files = glob.glob(os.path.join(VAL_DIR, "*"))
    if len(files) == 0:
        print(f"[ERROR] No images in {VAL_DIR}")
        return

    fig, axes = plt.subplots(1, 4, figsize=(20, 6))
    
    for i in range(4):
        sample = get_sample(random.choice(files))
        while sample is None: sample = get_sample(random.choice(files))
        pa, pb, gt = sample
        
        # Unsupervised Prediction
        stacked = np.stack([pa.astype(np.float32)/255., pb.astype(np.float32)/255.], axis=0)
        tensor = torch.from_numpy(stacked).unsqueeze(0).float().to(device)
        with torch.no_grad():
            pred_unsup = unsup_model(tensor).cpu().numpy().squeeze()
        
        # Classical Prediction
        pred_classic = get_classical_homography(pa, pb)

        # Plotting
        axes[i].imshow(pa, cmap='gray')
        plot_overlay(axes[i], gt, 'red', 'Ground Truth')
        plot_overlay(axes[i], pred_classic, 'lime', 'Classical', '--')
        plot_overlay(axes[i], pred_unsup, 'yellow', 'Unsupervised', '--')
        axes[i].set_title(f"Test Sample {i+1}")
        
        # Add legend only to the first plot
        if i == 0: 
            axes[i].legend(loc='lower right', fontsize='small')

    plt.tight_layout()
    save_path = "Fig18_Unsupervised_Comparisons.png"
    plt.savefig(save_path)
    print(f"Saved visualization to '{save_path}'")

    print("\n--- 4. QUANTITATIVE RESULTS (Unsupervised) ---")
    print("-" * 65)
    print(f"{'Dataset':10} | {'Metric':20} | {'Result'}")
    print("-" * 65)
    
    run_epe_benchmark("Train", unsup_model, TRAIN_DIR)
    run_epe_benchmark("Val", unsup_model, VAL_DIR)
    run_epe_benchmark("Test", unsup_model, TEST_DIR)
    print("-" * 65)

if __name__ == "__main__":
    main()