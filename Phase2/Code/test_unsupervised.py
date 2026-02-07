#!/usr/bin/env python3
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------
# Device
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# Model (same as training)
# ----------------------------
class HomographyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(2,64,kernel_size=3,padding=1,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,64,kernel_size=3,padding=1,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64,64,kernel_size=3,padding=1,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,64,kernel_size=3,padding=1,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64,128,kernel_size=3,padding=1,bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128,128,kernel_size=3,padding=1,bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(128,128,kernel_size=3,padding=1,bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128,128,kernel_size=3,padding=1,bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128*16*16, 1024),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(1024, 8)
        )

    def forward(self,x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.contiguous().view(x.size(0), -1)
        return self.fc(x)

class TensorDLT(nn.Module):
    def forward(self, corners_A, delta):
        B = corners_A.shape[0]
        corners_B = corners_A + delta

        u  = corners_A[..., 0]
        v  = corners_A[..., 1]
        up = corners_B[..., 0]
        vp = corners_B[..., 1]

        A = torch.zeros((B,8,8), device=corners_A.device, dtype=corners_A.dtype)
        b = torch.zeros((B,8,1), device=corners_A.device, dtype=corners_A.dtype)

        for i in range(4):
            A[:, 2*i,   3] = -u[:, i]
            A[:, 2*i,   4] = -v[:, i]
            A[:, 2*i,   5] = -1.0
            A[:, 2*i,   6] =  vp[:, i] * u[:, i]
            A[:, 2*i,   7] =  vp[:, i] * v[:, i]
            b[:, 2*i,   0] = -vp[:, i]

            A[:, 2*i+1, 0] =  u[:, i]
            A[:, 2*i+1, 1] =  v[:, i]
            A[:, 2*i+1, 2] =  1.0
            A[:, 2*i+1, 6] = -up[:, i] * u[:, i]
            A[:, 2*i+1, 7] = -up[:, i] * v[:, i]
            b[:, 2*i+1, 0] =  up[:, i]

        hhat = torch.linalg.lstsq(A, b).solution  # (B,8,1)

        H = torch.zeros((B,3,3), device=corners_A.device, dtype=corners_A.dtype)
        H[:,0,0] = hhat[:,0,0]; H[:,0,1] = hhat[:,1,0]; H[:,0,2] = hhat[:,2,0]
        H[:,1,0] = hhat[:,3,0]; H[:,1,1] = hhat[:,4,0]; H[:,1,2] = hhat[:,5,0]
        H[:,2,0] = hhat[:,6,0]; H[:,2,1] = hhat[:,7,0]; H[:,2,2] = 1.0
        return H

def warp_patch(patch_a, H, out_h=128, out_w=128):
    B, _, Ha, Wa = patch_a.shape
    dev = patch_a.device
    dtype = patch_a.dtype

    ys, xs = torch.meshgrid(
        torch.arange(out_h, device=dev, dtype=dtype),
        torch.arange(out_w, device=dev, dtype=dtype),
        indexing="ij"
    )
    ones = torch.ones_like(xs)
    pB = torch.stack([xs, ys, ones], dim=-1).view(1, out_h*out_w, 3).repeat(B,1,1)

    Hinv = torch.linalg.inv(H)
    pA = (Hinv @ pB.transpose(1,2)).transpose(1,2)

    den = pA[...,2].clamp(min=1e-8)
    xA = pA[...,0] / den
    yA = pA[...,1] / den

    xA_norm = (2.0 * xA / (Wa - 1)) - 1.0
    yA_norm = (2.0 * yA / (Ha - 1)) - 1.0
    grid = torch.stack([xA_norm, yA_norm], dim=-1).view(B, out_h, out_w, 2)

    return F.grid_sample(patch_a, grid, mode="bilinear", padding_mode="zeros", align_corners=True)

class UnsupervisedHomographyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.regressor = HomographyNet()
        self.dlt = TensorDLT()

    def forward(self, stacked, patch_a, corners_a, rho):
        pred8 = self.regressor(stacked)
        delta = (rho * torch.tanh(pred8)).view(-1,4,2)
        H = self.dlt(corners_a, delta)
        warped = warp_patch(patch_a, H, 128, 128)
        return warped, delta, H

# ----------------------------
# Data generation for testing
# ----------------------------
def make_pair_from_image(gray, crop=128, rho=32):
    H_img, W_img = gray.shape[:2]
    if W_img < crop + 2*rho or H_img < crop + 2*rho:
        raise ValueError("Image too small for given crop/rho")

    x = np.random.randint(rho, W_img - crop - rho)
    y = np.random.randint(rho, H_img - crop - rho)

    PA = gray[y:y+crop, x:x+crop]

    corners_img = np.array([[x, y],
                            [x+crop-1, y],
                            [x+crop-1, y+crop-1],
                            [x, y+crop-1]], dtype=np.float32)

    pert = np.random.randint(-(rho-1), (rho-1)+1, (4,2)).astype(np.float32)
    corners_pert = corners_img + pert

    H = cv2.getPerspectiveTransform(corners_img, corners_pert)
    warped_full = cv2.warpPerspective(gray, H, (W_img, H_img), flags=cv2.WARP_INVERSE_MAP)
    PB = warped_full[y:y+crop, x:x+crop]

    # patch-local corners
    corners_patch = np.array([[0,0],[crop-1,0],[crop-1,crop-1],[0,crop-1]], dtype=np.float32)

    # gt delta in patch coords (same as pert)
    gt_delta = pert.copy()
    return PA, PB, corners_patch, gt_delta

def save_vis(out_path, PA, PB, warped):
    # convert to uint8 for saving
    PAu = (PA*255).clip(0,255).astype(np.uint8)
    PBu = (PB*255).clip(0,255).astype(np.uint8)
    Wu  = (warped*255).clip(0,255).astype(np.uint8)
    # concat: PA | PB | warped(PA)
    vis = np.concatenate([PAu, PBu, Wu], axis=1)
    cv2.imwrite(out_path, vis)

# ----------------------------
# Main test
# ----------------------------
def test_once(model_path, image_path, out_dir="test_out", rho=32):
    os.makedirs(out_dir, exist_ok=True)

    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise RuntimeError(f"Failed to read image: {image_path}")

    PA, PB, corners_patch, gt_delta = make_pair_from_image(gray, crop=128, rho=rho)

    # to torch
    PA_t = torch.from_numpy((PA/255.0).astype(np.float32))[None,None].to(device)  # (1,1,128,128)
    PB_t = torch.from_numpy((PB/255.0).astype(np.float32))[None,None].to(device)

    stacked = torch.cat([PA_t, PB_t], dim=1)  # (1,2,128,128)
    corners = torch.from_numpy(corners_patch.astype(np.float32))[None].to(device)  # (1,4,2)

    model = UnsupervisedHomographyModel().to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    with torch.no_grad():
        warped, pred_delta, _ = model(stacked, PA_t, corners, rho=rho)

    # losses / metrics
    photometric = torch.mean(torch.abs(warped - PB_t)).item()

    pred_delta_np = pred_delta.squeeze(0).cpu().numpy()  # (4,2)
    rmse = np.sqrt(np.mean((pred_delta_np - gt_delta)**2))

    print(f"[TEST] Photometric L1: {photometric:.6f}")
    print(f"[TEST] Corner Delta RMSE (pixels): {rmse:.3f}")

    # save visualization
    warped_np = warped.squeeze().cpu().numpy()
    PA_np = (PA/255.0).astype(np.float32)
    PB_np = (PB/255.0).astype(np.float32)

    out_img = os.path.join(out_dir, "PA_PB_warped.png")
    save_vis(out_img, PA_np, PB_np, warped_np)
    print(f"[TEST] Saved visualization: {out_img}")

if __name__ == "__main__":
    MODEL_PATH = "/home/alien/YourDirectoryID_p1/Phase2/Code/unsupervised_homography_net.pth"
    IMAGE_PATH = "/home/alien/YourDirectoryID_p1/Phase1/Data/Train/CustomSet2/img5.jpeg"

    test_once(MODEL_PATH, IMAGE_PATH, out_dir="test_out", rho=32)
