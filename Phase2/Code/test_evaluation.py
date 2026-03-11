#!/usr/bin/env python
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import random
import argparse
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 1. REUSE MODEL ARCHITECTURE
# ==========================================
class HomographyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(2, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.Conv2d(64, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.MaxPool2d(2, 2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.Conv2d(64, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.MaxPool2d(2, 2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.Conv2d(128, 128, 3, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.MaxPool2d(2, 2)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.Conv2d(128, 128, 3, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(True)
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * 16 * 16, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 8)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.contiguous().view(x.size(0), -1)
        return self.fc(x)

class UnifiedHomographyModel(nn.Module):
    def __init__(self, model_type="Sup"):
        super().__init__()
        self.regressor = HomographyNet()
    def forward(self, stacked):
        return self.regressor(stacked)

# ==========================================
# 2. EVALUATION LOGIC
# ==========================================
def get_random_crop_and_perturb(image, crop_size=128, rho=32):
    H, W = image.shape
    if H < crop_size + 2*rho or W < crop_size + 2*rho:
        return None, None, None, None
        
    x = np.random.randint(rho, W - crop_size - rho)
    y = np.random.randint(rho, H - crop_size - rho)
    
    patch_a = image[y:y+crop_size, x:x+crop_size]
    
    corners_a = np.array([
        [x, y],
        [x+crop_size-1, y],
        [x+crop_size-1, y+crop_size-1],
        [x, y+crop_size-1]
    ], dtype=np.float32)
    
    perturbation = np.random.randint(-rho + 1, rho, (4, 2)).astype(np.float32)
    corners_b = corners_a + perturbation
    
    H_mat = cv2.getPerspectiveTransform(corners_a, corners_b)
    H_inv = np.linalg.inv(H_mat)
    
    warped_full = cv2.warpPerspective(image, H_inv, (W, H), flags=cv2.WARP_INVERSE_MAP)
    patch_b = warped_full[y:y+crop_size, x:x+crop_size]
    
    # FIX: Now returns 4 values to match the unpacking logic
    return patch_a, patch_b, perturbation, corners_a

def evaluate(test_path, model_path, num_samples=1000, rho=32):
    print(f"[INFO] Loading model from {model_path}...")
    model = UnifiedHomographyModel().to(device)
    
    if not os.path.exists(model_path):
        print(f"[ERROR] Model file not found: {model_path}")
        return

    # Robust Loading
    state = torch.load(model_path, map_location=device)
    try:
        model.regressor.load_state_dict(state)
    except:
        model.load_state_dict(state, strict=False)
    model.eval()

    files = glob.glob(os.path.join(test_path, "*"))
    if not files:
        print("[ERROR] No images found in test path.")
        return

    errors_mse = []
    
    print(f"[INFO] Running Inference on {num_samples} random pairs...")
    
    with torch.no_grad():
        for _ in tqdm(range(num_samples)):
            path = random.choice(files)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None: continue
            
            # This line caused the error before; now it works
            pa, pb, gt_delta, _ = get_random_crop_and_perturb(img, 128, rho)
            
            if pa is None: continue
            
            # Prepare Input
            stacked = np.stack([pa.astype(np.float32)/255.0, pb.astype(np.float32)/255.0], axis=0)
            tensor = torch.from_numpy(stacked).unsqueeze(0).float().to(device)
            
            # Inference
            pred_norm = model(tensor).cpu().numpy().flatten()
            pred_delta = pred_norm * float(rho)
            gt_flat = gt_delta.flatten()
            
            # Calculate Point-wise Error (Euclidean Distance per corner)
            diff = (pred_delta - gt_flat).reshape(4, 2)
            corner_errors = np.linalg.norm(diff, axis=1) # Shape (4,)
            
            # Mean Corner Error for this image
            mean_corner_error = np.mean(corner_errors)
            errors_mse.append(mean_corner_error)

    if len(errors_mse) == 0:
        print("[ERROR] No valid samples processed.")
        return

    # --- PLOTTING ---
    errors = np.array(errors_mse)
    mean_err = np.mean(errors)
    
    plt.figure(figsize=(14, 6))

    # PLOT 1: ERROR HISTOGRAM (Confusion Matrix Equivalent)
    plt.subplot(1, 2, 1)
    plt.hist(errors, bins=50, color='royalblue', edgecolor='black', alpha=0.7)
    plt.axvline(mean_err, color='red', linestyle='dashed', linewidth=2, label=f'Mean Error: {mean_err:.2f} px')
    plt.title(f"Test Error Distribution (N={len(errors)})")
    plt.xlabel("Average Corner Error (Pixels)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # PLOT 2: CUMULATIVE ACCURACY (CDF)
    plt.subplot(1, 2, 2)
    sorted_errors = np.sort(errors)
    yvals = np.arange(len(sorted_errors)) / float(len(sorted_errors))
    plt.plot(sorted_errors, yvals, color='green', linewidth=2)
    plt.title("Accuracy: Cumulative Error Distribution (CDF)")
    plt.xlabel("Corner Error Threshold (Pixels)")
    plt.ylabel("Fraction of Test Set")
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 20) # Focus on 0-20 pixel range
    plt.yticks(np.arange(0, 1.1, 0.1))

    # Save
    out_file = "test_accuracy_graphs.png"
    plt.savefig(out_file)
    print(f"\n[DONE] Graphs saved to {out_file}")
    print(f"       Average Test Error: {mean_err:.4f} pixels")
    # plt.show() # Commented out in case you are running on a headless server

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_path", default="/home/adipat/Documents/Spring_26/CV/P1/Deep_homography/Deep-Homography/Phase2/Data/Val", help="Path to Test/Val images")
    parser.add_argument("--model_path", default="/home/adipat/Documents/Spring_26/CV/P1/Deep_homography/Deep-Homography/model_weights/supervised_homography_net.pth", help="Path to .pth model")
    args = parser.parse_args()

    evaluate(args.test_path, args.model_path)