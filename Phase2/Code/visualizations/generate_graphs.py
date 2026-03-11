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

# --- CONFIGURATION (UPDATE THESE PATHS) ---
# Path to your Test/Val images for the 4-image comparison
VAL_DIR = "/home/adipat/Documents/Spring_26/CV/P1/Deep_homography/Deep-Homography/Phase2/Data/Val"

# Path to the Phase 2 Test Set (where Tower, Unity, Trees folders are)
TEST_SET_DIR = "/home/adipat/Documents/Spring_26/CV/P1/Deep_homography/Deep-Homography/Phase2/Data/P1Ph2TestSet/Phase2Pano"

# Model Paths
SUP_MODEL_PATH = "/home/adipat/Documents/Spring_26/CV/P1/Deep_homography/Deep-Homography/model_weights/supervised_homography_net.pth"
UNSUP_MODEL_PATH = "/home/adipat/Documents/Spring_26/CV/P1/Deep_homography/Deep-Homography/model_weights/unsupervised_homography_net.pth"

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
        self.fc = nn.Sequential(nn.Linear(128*16*16, 1024), nn.ReLU(True), nn.Linear(1024, 8)) # No dropout for inference

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
        if self.model_type == "Sup": return pred.view(-1, 4, 2) * float(rho)
        else: return (torch.tanh(pred) * float(rho)).view(-1, 4, 2)

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
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

def get_classical_homography(imgA, imgB):
    orb = cv2.ORB_create(nfeatures=2000)
    kp1, des1 = orb.detectAndCompute(imgA, None)
    kp2, des2 = orb.detectAndCompute(imgB, None)
    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4: return np.zeros((4, 2))
    
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(bf.match(des1, des2), key=lambda x: x.distance)[:50]
    if len(matches) < 4: return np.zeros((4, 2))

    src = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    if H is None: return np.zeros((4, 2))

    base = np.array([[0,0], [128,0], [128,128], [0,128]], dtype=np.float32).reshape(-1, 1, 2)
    est = cv2.perspectiveTransform(base, H)
    return est.reshape(4, 2) - base.reshape(4, 2)

def plot_overlay(ax, delta, color, label, linestyle='-'):
    base = np.array([[0,0], [128,0], [128,128], [0,128]], dtype=np.float32)
    poly = base + delta
    poly = np.vstack([poly, poly[0]])
    ax.plot(poly[:,0], poly[:,1], color=color, linewidth=2, linestyle=linestyle, label=label)

# ==========================================
# 3. TASK 1: RUNTIME BENCHMARK
# ==========================================
def benchmark_model(model, name="Model"):
    dummy_input = torch.randn(1, 2, 128, 128).to(device)
    
    # Warmup
    for _ in range(10): _ = model(dummy_input)
    
    # Timing
    start = time.time()
    for _ in range(100):
        _ = model(dummy_input)
        if torch.cuda.is_available(): torch.cuda.synchronize()
    end = time.time()
    
    avg_time = (end - start) * 1000 / 100.0
    print(f"✅ {name} Avg Runtime: {avg_time:.4f} ms")

# ==========================================
# 4. TASK 2: 4-IMAGE COMPARISON
# ==========================================
def generate_comparison_figure(sup_model, unsup_model):
    print("\n[INFO] Generating 4-Image Comparison Figure...")
    files = glob.glob(os.path.join(VAL_DIR, "*"))
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 6))
    
    for i in range(4):
        sample = None
        while sample is None: sample = get_sample(random.choice(files))
        pa, pb, gt = sample
        
        # Prepare Input
        stacked = np.stack([pa.astype(np.float32)/255., pb.astype(np.float32)/255.], axis=0)
        tensor = torch.from_numpy(stacked).unsqueeze(0).float().to(device)
        
        # Inference
        with torch.no_grad():
            pred_sup = sup_model(tensor).cpu().numpy().squeeze()
            pred_unsup = unsup_model(tensor).cpu().numpy().squeeze()
        
        pred_classic = get_classical_homography(pa, pb)

        # Plot
        axes[i].imshow(pa, cmap='gray')
        plot_overlay(axes[i], gt, 'red', 'Ground Truth')
        plot_overlay(axes[i], pred_classic, 'lime', 'Classical', ':')
        plot_overlay(axes[i], pred_sup, 'yellow', 'Supervised', '--')
        plot_overlay(axes[i], pred_unsup, 'cyan', 'Unsupervised', '-.')
        
        axes[i].set_title(f"Sample {i+1}")
        if i == 0: axes[i].legend(loc='lower right')

    plt.tight_layout()
    plt.savefig("Phase2_4_Comparisons.png")
    print("✅ Saved 'Phase2_4_Comparisons.png'")

# ==========================================
# 5. TASK 3: INPUT STRIPS
# ==========================================
def generate_input_strips():
    print("\n[INFO] Generating Input Strips for Phase 2 Test Sets...")
    
    sets = ["Set1", "Set2", "Set3"]  # Usually correspond to Tower, Unity, Trees
    
    for s in sets:
        folder_path = os.path.join(TEST_SET_DIR, s)
        if not os.path.exists(folder_path):
            print(f"[WARN] Folder {s} not found in {TEST_SET_DIR}")
            continue
            
        images = sorted(glob.glob(os.path.join(folder_path, "*.jpg")) + glob.glob(os.path.join(folder_path, "*.png")))
        if not images: continue
        
        # Take first 5 images or fewer
        subset = images[:5]
        loaded_imgs = []
        for p in subset:
            img = cv2.imread(p)
            # Resize for strip (height 150, maintain aspect ratio)
            h, w = img.shape[:2]
            new_h = 150
            new_w = int(w * (new_h / h))
            img = cv2.resize(img, (new_w, new_h))
            loaded_imgs.append(img)
            
        # Concatenate horizontally
        strip = np.hstack(loaded_imgs)
        out_name = f"Input_Strip_{s}.png"
        cv2.imwrite(out_name, strip)
        print(f"✅ Saved '{out_name}' (Input for {s})")

def main():
    # Load Models
    print("--- Loading Models ---")
    sup_model = UnifiedHomographyModel("Sup").to(device)
    unsup_model = UnifiedHomographyModel("Unsup").to(device)
    
    try:
        sup_model.regressor.load_state_dict(torch.load(SUP_MODEL_PATH, map_location=device))
        unsup_model.regressor.load_state_dict(torch.load(UNSUP_MODEL_PATH, map_location=device))
    except:
        # Fallback for strict loading issues
        sup_model.load_state_dict(torch.load(SUP_MODEL_PATH, map_location=device), strict=False)
        unsup_model.load_state_dict(torch.load(UNSUP_MODEL_PATH, map_location=device), strict=False)
        
    sup_model.eval()
    unsup_model.eval()

    # 1. Benchmark
    print("\n--- 1. Runtime Benchmark ---")
    benchmark_model(sup_model, "Supervised")
    benchmark_model(unsup_model, "Unsupervised")
    
    # 2. Comparison Figure
    generate_comparison_figure(sup_model, unsup_model)
    
    # 3. Input Strips
    generate_input_strips()

if __name__ == "__main__":
    main()
    