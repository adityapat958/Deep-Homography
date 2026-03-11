#!/usr/bin/env python
import torch
import torch.nn as nn
import numpy as np
import cv2
import os
import glob
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
# UPDATE THESE PATHS TO MATCH YOUR SYSTEM EXACTLY

DATA_DIR = "/home/adipat/Documents/Spring 26/CV/P1/Deep_homography/Deep-Homography/Phase2/Data/Train_Phase1/Set1" 
MODEL_DIR = "/home/adipat/Documents/Spring 26/CV/P1/Deep_homography/Deep-Homography/model_weights" 

# Ensure these match the filenames you saved
SUP_MODEL_NAME = "Sup_homography_net.pth"
UNSUP_MODEL_NAME = "unsupervised_homography_net.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 1. MODEL DEFINITIONS 
# ==========================================
class HomographyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(2, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True)
        )
        self.fc = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(128 * 16 * 16, 1024),
            nn.ReLU(True),
            nn.Dropout(0.1),
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
        self.model_type = model_type

    def forward(self, stacked, rho=32):
        pred8 = self.regressor(stacked)
        if self.model_type == "Sup":
            delta = pred8.view(-1, 4, 2) * float(rho)
        else:
            delta = (torch.tanh(pred8) * float(rho)).view(-1, 4, 2)
        return delta

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def load_model(path, model_type):
    print(f"[INFO] Loading {model_type} model from {path}...")
    model = UnifiedHomographyModel(model_type=model_type).to(device)
    
    if not os.path.exists(path):
        print(f"[ERROR] Path does not exist: {path}")
        return None

    state = torch.load(path, map_location=device)
    
    # Try loading regressor weights (handle different saving formats)
    try:
        model.regressor.load_state_dict(state)
    except:
        try:
            model.load_state_dict(state)
        except:
            print(f"[WARN] Standard loading failed for {model_type}, trying strict=False")
            model.load_state_dict(state, strict=False)
    
    model.eval()
    return model

def get_random_sample(path, crop_size=128, rho=32):
    """ Creates a synthetic training pair (Patch A -> Patch B) """
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None: return None
    
    H, W = img.shape
    if H < crop_size + 2*rho or W < crop_size + 2*rho: return None

    x = np.random.randint(rho, W - crop_size - rho)
    y = np.random.randint(rho, H - crop_size - rho)

    # Patch A
    patch_a = img[y:y+crop_size, x:x+crop_size]
    
    # GT Perturbation
    corners_a = np.array([[0, 0], [crop_size, 0], [crop_size, crop_size], [0, crop_size]], dtype=np.float32)
    
    # --- FIX: Explicitly cast everything to float32 for OpenCV ---
    global_corners_a = (corners_a + np.array([x, y], dtype=np.float32)).astype(np.float32)
    
    perturbation = np.random.randint(-rho + 1, rho, (4, 2)).astype(np.float32)
    global_corners_b = (global_corners_a + perturbation).astype(np.float32)
    
    # This line was crashing because inputs were float64
    H_mat = cv2.getPerspectiveTransform(global_corners_a, global_corners_b)
    H_inv = np.linalg.inv(H_mat)
    
    warped_full = cv2.warpPerspective(img, H_inv, (W, H), flags=cv2.WARP_INVERSE_MAP)
    patch_b = warped_full[y:y+crop_size, x:x+crop_size]

    return patch_a, patch_b, perturbation

# ==========================================
# 3. VISUALIZATION
# ==========================================
def draw_box(ax, corners, color, label, linestyle='-'):
    """ Draws a polygon on the matplotlib axis """
    # Close the loop
    plot_corners = np.vstack([corners, corners[0]])
    ax.plot(plot_corners[:, 0], plot_corners[:, 1], color=color, linewidth=2, linestyle=linestyle, label=label)

def main():
    # 1. Load Models
    path_sup = os.path.join(MODEL_DIR, SUP_MODEL_NAME)
    path_unsup = os.path.join(MODEL_DIR, UNSUP_MODEL_NAME)
    
    model_sup = load_model(path_sup, "Sup")
    model_unsup = load_model(path_unsup, "Unsup")

    if model_sup is None or model_unsup is None:
        print("[ERROR] Failed to load one or both models. Exiting.")
        return

    # 2. Get Data
    images = glob.glob(os.path.join(DATA_DIR, "*.jpg")) + glob.glob(os.path.join(DATA_DIR, "*.png"))
    if not images:
        print(f"[ERROR] No images found in {DATA_DIR}")
        return
        
    sample = None
    while sample is None:
        path = np.random.choice(images)
        sample = get_random_sample(path)
    
    pa, pb, gt_delta = sample
    
    # 3. Prepare Input
    pa_norm = pa.astype(np.float32) / 255.0
    pb_norm = pb.astype(np.float32) / 255.0
    stacked = np.stack([pa_norm, pb_norm], axis=0)
    tensor = torch.from_numpy(stacked).unsqueeze(0).float().to(device)

    # 4. Predict
    with torch.no_grad():
        delta_sup = model_sup(tensor, rho=32).cpu().numpy().squeeze()
        delta_unsup = model_unsup(tensor, rho=32).cpu().numpy().squeeze()

    # 5. Plotting
    corners_a = np.array([[0, 0], [128, 0], [128, 128], [0, 128]], dtype=np.float32)
    corners_gt = corners_a + gt_delta
    corners_sup = corners_a + delta_sup
    corners_unsup = corners_a + delta_unsup

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # --- PLOT 1: SUPERVISED ---
    axes[0].imshow(pa, cmap='gray')
    axes[0].set_title("Supervised Model")
    draw_box(axes[0], corners_gt, 'blue', 'Ground Truth')
    draw_box(axes[0], corners_sup, 'yellow', 'Predicted', linestyle='--')
    axes[0].legend(loc='lower right')

    # --- PLOT 2: UNSUPERVISED ---
    axes[1].imshow(pa, cmap='gray')
    axes[1].set_title("Unsupervised Model")
    draw_box(axes[1], corners_gt, 'blue', 'Ground Truth')
    draw_box(axes[1], corners_unsup, 'yellow', 'Predicted', linestyle='--')
    axes[1].legend(loc='lower right')

    plt.tight_layout()
    plt.show()
    print("[DONE] Visualization displayed.")

if __name__ == "__main__":
    main()