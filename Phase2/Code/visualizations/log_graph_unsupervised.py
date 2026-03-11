#!/usr/bin/env python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import os

# --- CONFIGURATION ---
LOG_DIR = "Logs/Unsup_Architecture_Corrected" 

# ==========================================
# 1. DEFINE FULL UNSUPERVISED PIPELINE
# ==========================================

class HomographyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(2,64,3,padding=1,bias=False), nn.BatchNorm2d(64), nn.ReLU(True), nn.Conv2d(64,64,3,padding=1,bias=False), nn.BatchNorm2d(64), nn.ReLU(True), nn.MaxPool2d(2,2))
        self.layer2 = nn.Sequential(nn.Conv2d(64,64,3,padding=1,bias=False), nn.BatchNorm2d(64), nn.ReLU(True), nn.Conv2d(64,64,3,padding=1,bias=False), nn.BatchNorm2d(64), nn.ReLU(True), nn.MaxPool2d(2,2))
        self.layer3 = nn.Sequential(nn.Conv2d(64,128,3,padding=1,bias=False), nn.BatchNorm2d(128), nn.ReLU(True), nn.Conv2d(128,128,3,padding=1,bias=False), nn.BatchNorm2d(128), nn.ReLU(True), nn.MaxPool2d(2,2))
        self.layer4 = nn.Sequential(nn.Conv2d(128,128,3,padding=1,bias=False), nn.BatchNorm2d(128), nn.ReLU(True), nn.Conv2d(128,128,3,padding=1,bias=False), nn.BatchNorm2d(128), nn.ReLU(True))
        self.fc = nn.Sequential(nn.Dropout(0.1), nn.Linear(128*16*16, 1024), nn.ReLU(True), nn.Dropout(0.1), nn.Linear(1024, 8))

    def forward(self, x):
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        return self.fc(x.contiguous().view(x.size(0), -1))

class TensorDLT(nn.Module):
    def forward(self, corners_A, delta):
        B = corners_A.shape[0]
        corners_B = corners_A + delta
        u, v = corners_A[..., 0], corners_A[..., 1]
        up, vp = corners_B[..., 0], corners_B[..., 1]

        A = torch.zeros((B, 8, 8), device=corners_A.device)
        b = torch.zeros((B, 8, 1), device=corners_A.device)

        for i in range(4):
            A[:, 2*i, 3] = -u[:, i]; A[:, 2*i, 4] = -v[:, i]; A[:, 2*i, 5] = -1.0
            A[:, 2*i, 6] = vp[:, i] * u[:, i]; A[:, 2*i, 7] = vp[:, i] * v[:, i]
            b[:, 2*i, 0] = -vp[:, i]
            A[:, 2*i+1, 0] = u[:, i]; A[:, 2*i+1, 1] = v[:, i]; A[:, 2*i+1, 2] = 1.0
            A[:, 2*i+1, 6] = -up[:, i] * u[:, i]; A[:, 2*i+1, 7] = -up[:, i] * v[:, i]
            b[:, 2*i+1, 0] = up[:, i]

        hhat = torch.linalg.lstsq(A, b).solution
        H = torch.zeros((B, 3, 3), device=corners_A.device)
        H[:, 0, 0] = hhat[:, 0, 0]; H[:, 0, 1] = hhat[:, 1, 0]; H[:, 0, 2] = hhat[:, 2, 0]
        H[:, 1, 0] = hhat[:, 3, 0]; H[:, 1, 1] = hhat[:, 4, 0]; H[:, 1, 2] = hhat[:, 5, 0]
        H[:, 2, 0] = hhat[:, 6, 0]; H[:, 2, 1] = hhat[:, 7, 0]; H[:, 2, 2] = 1.0
        return H

class UnifiedHomographyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.regressor = HomographyNet()
        self.dlt = TensorDLT()

    def forward(self, stacked_input):
        stacked, patch_a, corners_a = stacked_input
        
        # 1. Regress
        pred8 = self.regressor(stacked)
        delta = (32.0 * torch.tanh(pred8)).view(-1, 4, 2)
        
        # 2. DLT
        H = self.dlt(corners_a, delta)
        
        # 3. WARPING (Manual STN Implementation)
        # This correctly handles the Perspective Division
        B, C, H_in, W_in = patch_a.shape
        device = patch_a.device
        
        # Generate Target Meshgrid (x,y)
        ys, xs = torch.meshgrid(torch.arange(H_in, device=device), torch.arange(W_in, device=device), indexing='ij')
        # Flatten and append 1s for homogeneous coords
        ones = torch.ones_like(xs)
        coords = torch.stack([xs, ys, ones], dim=-1).view(1, H_in * W_in, 3).repeat(B, 1, 1).float() # (B, N, 3)
        
        # Inverse Warp: Source = H_inv * Target
        H_inv = torch.linalg.inv(H)
        src_coords = torch.bmm(H_inv, coords.transpose(1, 2)).transpose(1, 2) # (B, N, 3)
        
        # Perspective Division (The critical step I missed in v1)
        x_src = src_coords[..., 0] / (src_coords[..., 2] + 1e-8)
        y_src = src_coords[..., 1] / (src_coords[..., 2] + 1e-8)
        
        # Normalize to [-1, 1] for grid_sample
        norm_x = (2.0 * x_src / (W_in - 1)) - 1.0
        norm_y = (2.0 * y_src / (H_in - 1)) - 1.0
        
        grid = torch.stack([norm_x, norm_y], dim=-1).view(B, H_in, W_in, 2)
        
        # Sample
        warped_a = F.grid_sample(patch_a, grid, align_corners=True)
        
        return warped_a, delta

def main():
    model = UnifiedHomographyModel()
    
    # Dummy Inputs
    dummy_stacked = torch.randn(1, 2, 128, 128)
    dummy_patch_a = torch.randn(1, 1, 128, 128)
    dummy_corners = torch.tensor([[[0., 0.], [128., 0.], [128., 128.], [0., 128.]]])
    
    input_tuple = (dummy_stacked, dummy_patch_a, dummy_corners)

    print(f"[INFO] Generating Corrected Unsupervised graph log in {LOG_DIR}...")
    
    # Clear old log if exists
    import shutil
    if os.path.exists(LOG_DIR): shutil.rmtree(LOG_DIR)
        
    writer = SummaryWriter(LOG_DIR)
    writer.add_graph(model, (input_tuple, ))
    writer.close()
    
    print("[DONE] Graph saved.")
    print(f"Run: tensorboard --logdir {LOG_DIR}")

if __name__ == "__main__":
    main()