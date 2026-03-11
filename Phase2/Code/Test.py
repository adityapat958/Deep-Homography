#!/usr/bin/env python
import numpy as np
import cv2
import os 
import glob
import torch
import torch.nn as nn
from tqdm import tqdm 

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 1. NETWORK (Must Match Train.py)
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
            nn.Dropout(0.5),
            nn.Linear(128 * 16 * 16, 1024),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(1024, 8)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.contiguous().view(x.size(0), -1)
        return self.fc(x)

# ==========================================
# 2. STITCHING UTILS
# ==========================================

def cylindrical_warp(img, f):
    h, w = img.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    ys, xs = np.indices((h, w), dtype=np.float32)
    x = xs - cx
    y = ys - cy
    theta = np.arctan2(x, f)
    x_c = f * theta
    y_c = f * y / np.sqrt(x * x + f * f)
    map_x = (x_c + cx).astype(np.float32)
    map_y = (y_c + cy).astype(np.float32)
    warped = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    # Mask
    mask = np.zeros((h, w), dtype=np.uint8)
    valid = (map_x >= 0) & (map_x < w) & (map_y >= 0) & (map_y < h)
    mask[valid] = 255
    return warped, mask

def extract_patch_center(gray, cx, cy, crop):
    half = crop // 2
    x0 = int(round(cx)) - half
    y0 = int(round(cy)) - half
    return x0, y0, gray[y0:y0+crop, x0:x0+crop]

def valid_patch(x0, y0, H, W, crop, margin):
    return (x0 >= margin and y0 >= margin and x0 + crop < (W - margin) and y0 + crop < (H - margin))

def compute_pairwise_H_from_matches(model, imgA, imgB, crop=128, rho=32, margin=32, 
                                    max_matches=300, ransac_thresh=4.0, use_affine=False, model_type="Sup"):
    grayA = cv2.cvtColor(imgA, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(imgB, cv2.COLOR_BGR2GRAY)

    # 1. ORB Detection
    orb = cv2.ORB_create(nfeatures=5000)
    kA, dA = orb.detectAndCompute(grayA, None)
    kB, dB = orb.detectAndCompute(grayB, None)
    if dA is None or dB is None or len(kA) < 20 or len(kB) < 20: return np.eye(3, dtype=np.float32)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(dA, dB)
    matches = sorted(matches, key=lambda m: m.distance)[:max_matches]
    if len(matches) < 20: return np.eye(3, dtype=np.float32)

    # 2. Extract Patches
    corners_patch = np.array([[0, 0], [crop-1, 0], [crop-1, crop-1], [0, crop-1]], dtype=np.float32)
    stacked_list = []
    meta = []

    for m in matches:
        xa, ya = kA[m.queryIdx].pt
        xb, yb = kB[m.trainIdx].pt
        x0A, y0A, PA = extract_patch_center(grayA, xa, ya, crop)
        x0B, y0B, PB = extract_patch_center(grayB, xb, yb, crop)

        if not valid_patch(x0A, y0A, grayA.shape[0], grayA.shape[1], crop, margin): continue
        if not valid_patch(x0B, y0B, grayB.shape[0], grayB.shape[1], crop, margin): continue
        if PA.std() < 5 or PB.std() < 5: continue

        PA = PA.astype(np.float32) / 255.0
        PB = PB.astype(np.float32) / 255.0
        stacked_list.append(np.stack([PA, PB], axis=0))
        meta.append((x0A, y0A, x0B, y0B))

    if len(stacked_list) < 8: return np.eye(3, dtype=np.float32)

    # 3. Inference
    stacked = torch.from_numpy(np.stack(stacked_list, axis=0)).float().to(device)
    model.eval()
    with torch.no_grad():
        pred = model(stacked) 
        
        if model_type == "Sup":
            # Sup trained with output / rho -> recover by * rho
            delta = pred.view(-1, 4, 2) * float(rho)
        else:
            # Unsup trained with tanh -> recover by * rho
            delta = (torch.tanh(pred) * float(rho)).view(-1, 4, 2)
            
    delta = delta.cpu().numpy()

    # 4. Global H/Affine
    ptsA, ptsB = [], []
    for i, (x0A, y0A, x0B, y0B) in enumerate(meta):
        cA = corners_patch + np.array([x0A, y0A], dtype=np.float32)
        cB = (corners_patch + delta[i]) + np.array([x0A, y0A], dtype=np.float32) # Anchor to A for relative shift
        ptsA.append(cA)
        ptsB.append(cB)

    ptsA = np.vstack(ptsA).astype(np.float32)
    ptsB = np.vstack(ptsB).astype(np.float32)

    if use_affine:
        M, inl = cv2.estimateAffinePartial2D(ptsA, ptsB, method=cv2.RANSAC, ransacReprojThreshold=ransac_thresh)
        if M is None: return np.eye(3, dtype=np.float32)
        H_AB = np.eye(3, dtype=np.float32)
        H_AB[:2, :] = M
    else:
        H_AB, inl = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, ransacReprojThreshold=ransac_thresh)
        if H_AB is None: return np.eye(3, dtype=np.float32)

    return (H_AB / H_AB[2, 2]).astype(np.float32)

def feather_weight(mask):
    mask = (mask > 0).astype(np.uint8)
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    if dist.max() < 1e-6: return dist.astype(np.float32)
    w = dist / (dist.max() + 1e-6)
    w = cv2.GaussianBlur(w, (51, 51), 0)
    w = w / (w.max() + 1e-6)
    return w.astype(np.float32)

def warp_with_mask(img, H, out_w, out_h):
    warped = cv2.warpPerspective(img, H, (out_w, out_h), flags=cv2.INTER_LINEAR)
    base_mask = np.ones((img.shape[0], img.shape[1]), np.uint8) * 255
    wmask = cv2.warpPerspective(base_mask, H, (out_w, out_h), flags=cv2.INTER_NEAREST)
    return warped, wmask

def stitch_folder(folder_path, model_path, out_path, model_type="Sup", video=False, skip=8):
    torch.cuda.empty_cache() # Cleanup
    
    # Load Images
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    paths = []
    for e in exts: paths.extend(glob.glob(os.path.join(folder_path, e)))
    paths = sorted(list(set(paths)))

    # Video Toggle Logic
    if video:
        print(f"   [VIDEO MODE] Skipping {skip} frames.")
        paths = paths[::skip]
        focal_scale = 10.0 # Planar
        use_affine = True
        ransac = 2.0
    else:
        focal_scale = 1.0 # Cylindrical
        use_affine = False
        ransac = 4.0

    if len(paths) < 2: 
        print(f"[SKIP] Not enough images in {folder_path}"); return

    imgs = [cv2.imread(p) for p in paths if cv2.imread(p) is not None]
    
    # Pre-warp (Cylindrical or Identity)
    f = float(focal_scale * imgs[0].shape[1])
    cyl_imgs = []
    for im in imgs:
        if video: cyl_imgs.append(im) # No warp for video usually better
        else:
            wim, _ = cylindrical_warp(im, f)
            cyl_imgs.append(wim)

    # Load Model
    model = HomographyNet().to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # Canvas
    h, w = cyl_imgs[0].shape[:2]
    canvas_h, canvas_w = int(h*2.5), int(w*5.0)
    
    # Anchor
    mid = len(cyl_imgs) // 2
    H_global = [np.eye(3, dtype=np.float32) for _ in range(len(cyl_imgs))]
    H_global[mid] = np.array([[1, 0, canvas_w//2 - w//2], [0, 1, canvas_h//2 - h//2], [0, 0, 1]], dtype=np.float32)

    # Right
    print(f"   Stitching Right from {mid}...")
    for i in range(mid + 1, len(cyl_imgs)):
        H_rel = compute_pairwise_H_from_matches(model, cyl_imgs[i-1], cyl_imgs[i], 
                                                use_affine=use_affine, ransac_thresh=ransac, model_type=model_type)
        try: H_inv = np.linalg.inv(H_rel)
        except: H_inv = np.eye(3)
        H_global[i] = H_global[i-1] @ H_inv
        H_global[i] /= H_global[i][2, 2]

    # Left
    print(f"   Stitching Left from {mid}...")
    for i in range(mid - 1, -1, -1):
        H_rel = compute_pairwise_H_from_matches(model, cyl_imgs[i], cyl_imgs[i+1], 
                                                use_affine=use_affine, ransac_thresh=ransac, model_type=model_type)
        H_global[i] = H_global[i+1] @ H_rel
        H_global[i] /= H_global[i][2, 2]

    # Blend
    print("   Blending...")
    acc = np.zeros((canvas_h, canvas_w, 3), dtype=np.float32)
    accw = np.zeros((canvas_h, canvas_w), dtype=np.float32)
    
    for i in range(len(cyl_imgs)):
        wimg, m = warp_with_mask(cyl_imgs[i], H_global[i], canvas_w, canvas_h)
        w = feather_weight(m)
        acc += wimg * w[..., None]
        accw += w

    pano = (acc / np.clip(accw[..., None], 1e-6, None)).clip(0, 255).astype(np.uint8)
    
    # Auto Crop
    gray = cv2.cvtColor(pano, cv2.COLOR_BGR2GRAY)
    ys, xs = np.where(gray > 0)
    if len(xs) > 0:
        pano = pano[ys.min():ys.max(), xs.min():xs.max()]
        
    cv2.imwrite(out_path, pano)
    print(f"[DONE] Saved {out_path}")

# ==========================================
# 3. MAIN
# ==========================================

if __name__ == "__main__":
    # --- TOGGLES ---
    VIDEO_MODE = True       # Set True for Video folders, False for Image sets
    MODEL_TYPE = "Unsup"    # "Sup" or "Unsup" (Must match loaded checkpoint)
    
    # PATHS
    ROOT_DIR = "/home/adipat/Documents/Spring 26/CV/P1/Deep_homography/Deep-Homography/Phase2/Data/P1Ph2TestSet/Phase2Pano"
    MODEL_PATH = "/home/adipat/Documents/Spring 26/CV/P1/Deep_homography/Deep-Homography/unsupervised_homography_net.pth"
    OUTPUT_DIR = "/home/adipat/Documents/Spring 26/CV/P1/Deep_homography/Deep-Homography/Phase2/Data/results"

    if not os.path.exists(MODEL_PATH): print("Model not found!"); exit()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    subfolders = [f.path for f in os.scandir(ROOT_DIR) if f.is_dir()]
    print(f"Found {len(subfolders)} folders.")

    for folder in subfolders:
        name = os.path.basename(folder)
        print(f"\n--- Processing {name} ---")
        
        save_dir = os.path.join(OUTPUT_DIR, name)
        os.makedirs(save_dir, exist_ok=True)
        out_path = os.path.join(save_dir, "mypano.png")
        
        try:
            stitch_folder(folder, MODEL_PATH, out_path, 
                          model_type=MODEL_TYPE, 
                          video=VIDEO_MODE, 
                          skip=8 if VIDEO_MODE else 1)
        except Exception as e:
            print(f"[ERROR] {e}")