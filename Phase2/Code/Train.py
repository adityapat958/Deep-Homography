#!/usr/bin/env python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import cv2
import os
import numpy as np
import random
import glob
import argparse
from tqdm import tqdm
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 1. NETWORK ARCHITECTURE (Unified)
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
        # NO DROPOUT for Supervised Regression
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

class TensorDLT(nn.Module):
    def forward(self, corners_A, delta):
        B = corners_A.shape[0]
        corners_B = corners_A + delta
        u, v = corners_A[..., 0], corners_A[..., 1]
        up, vp = corners_B[..., 0], corners_B[..., 1]

        A = torch.zeros((B, 8, 8), device=corners_A.device)
        b = torch.zeros((B, 8, 1), device=corners_A.device)

        for i in range(4):
            A[:, 2*i, 3] = -u[:, i]
            A[:, 2*i, 4] = -v[:, i]
            A[:, 2*i, 5] = -1.0
            A[:, 2*i, 6] = vp[:, i] * u[:, i]
            A[:, 2*i, 7] = vp[:, i] * v[:, i]
            b[:, 2*i, 0] = -vp[:, i]

            A[:, 2*i+1, 0] = u[:, i]
            A[:, 2*i+1, 1] = v[:, i]
            A[:, 2*i+1, 2] = 1.0
            A[:, 2*i+1, 6] = -up[:, i] * u[:, i]
            A[:, 2*i+1, 7] = -up[:, i] * v[:, i]
            b[:, 2*i+1, 0] = up[:, i]

        hhat = torch.linalg.lstsq(A, b).solution
        H = torch.zeros((B, 3, 3), device=corners_A.device)
        H[:, 0, 0] = hhat[:, 0, 0]; H[:, 0, 1] = hhat[:, 1, 0]; H[:, 0, 2] = hhat[:, 2, 0]
        H[:, 1, 0] = hhat[:, 3, 0]; H[:, 1, 1] = hhat[:, 4, 0]; H[:, 1, 2] = hhat[:, 5, 0]
        H[:, 2, 0] = hhat[:, 6, 0]; H[:, 2, 1] = hhat[:, 7, 0]; H[:, 2, 2] = 1.0
        return H

def warp_patch(patch_a, H, out_h=128, out_w=128):
    B = patch_a.shape[0]
    dev = patch_a.device
    ys, xs = torch.meshgrid(torch.arange(out_h, device=dev), torch.arange(out_w, device=dev), indexing="ij")
    ones = torch.ones_like(xs)
    pB = torch.stack([xs, ys, ones], dim=-1).view(1, out_h * out_w, 3).repeat(B, 1, 1).float()
    
    Hinv = torch.linalg.inv(H)
    pA = (Hinv @ pB.transpose(1, 2)).transpose(1, 2)
    den = pA[..., 2].clamp(min=1e-8)
    xA, yA = pA[..., 0] / den, pA[..., 1] / den
    
    grid = torch.stack([(2.0 * xA / (out_w - 1)) - 1.0, (2.0 * yA / (out_h - 1)) - 1.0], dim=-1).view(B, out_h, out_w, 2)
    return F.grid_sample(patch_a, grid, mode="bilinear", padding_mode="zeros", align_corners=True)

class UnifiedHomographyModel(nn.Module):
    def __init__(self, model_type="Sup"):
        super().__init__()
        self.regressor = HomographyNet()
        self.dlt = TensorDLT()
        self.model_type = model_type

    def forward(self, stacked, patch_a=None, corners_a=None, rho=32):
        pred8 = self.regressor(stacked)
        
        if self.model_type == "Sup":
            return pred8
        else:
            delta = (rho * torch.tanh(pred8)).view(-1, 4, 2)
            H = self.dlt(corners_a, delta)
            warped_a = warp_patch(patch_a, H)
            return warped_a, delta

# ==========================================
# 2. DATA GENERATION (Manual Batching)
# ==========================================
def get_random_crop_and_perturb(image, crop_size=128, rho=32):
    H, W = image.shape
    if H < crop_size + 2*rho or W < crop_size + 2*rho: return None, None, None, None
    
    x = np.random.randint(rho, W - crop_size - rho)
    y = np.random.randint(rho, H - crop_size - rho)
    
    patch_a = image[y:y+crop_size, x:x+crop_size]
    corners_a = np.array([[x, y], [x+crop_size-1, y], [x+crop_size-1, y+crop_size-1], [x, y+crop_size-1]], dtype=np.float32)
    
    perturbation = np.random.randint(-rho + 1, rho, (4, 2)).astype(np.float32)
    corners_b = corners_a + perturbation
    
    H_mat = cv2.getPerspectiveTransform(corners_a, corners_b)
    H_inv = np.linalg.inv(H_mat)
    
    warped_full = cv2.warpPerspective(image, H_inv, (W, H), flags=cv2.WARP_INVERSE_MAP)
    patch_b = warped_full[y:y+crop_size, x:x+crop_size]
    
    return patch_a, patch_b, perturbation, corners_a

def GenerateBatch(FileList, BatchSize, Rho=32, ModelType="Sup"):
    stacked_batch = []
    label_batch = []
    patch_a_batch = []
    patch_b_batch = []
    corners_batch = []
    
    count = 0
    while count < BatchSize:
        path = random.choice(FileList) # Path is already full path
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None: continue
        
        pa, pb, delta, _ = get_random_crop_and_perturb(img, 128, Rho)
        if pa is None: continue
        
        # Normalize
        stacked = np.stack([pa.astype(np.float32)/255.0, pb.astype(np.float32)/255.0], axis=0)
        
        if ModelType == "Sup":
            norm_delta = delta.flatten() / float(Rho)
            stacked_batch.append(torch.from_numpy(stacked))
            label_batch.append(torch.from_numpy(norm_delta))
        else:
            # For Unsup we need warped images and local corners
            corners_local = np.array([[0, 0], [127, 0], [127, 127], [0, 127]], dtype=np.float32)
            stacked_batch.append(torch.from_numpy(stacked))
            patch_a_batch.append(torch.from_numpy(pa.astype(np.float32)/255.0).unsqueeze(0))
            patch_b_batch.append(torch.from_numpy(pb.astype(np.float32)/255.0).unsqueeze(0))
            corners_batch.append(torch.from_numpy(corners_local))
            
        count += 1
    
    inputs = torch.stack(stacked_batch).to(device)
    
    if ModelType == "Sup":
        return {"input": inputs, "label": torch.stack(label_batch).to(device)}
    else:
        return {
            "input": inputs, 
            "patch_a": torch.stack(patch_a_batch).to(device),
            "patch_b": torch.stack(patch_b_batch).to(device),
            "corners": torch.stack(corners_batch).to(device)
        }

# ==========================================
# 3. TRAINING & LOGGING
# ==========================================
def calculate_rmse(preds_norm, labels_norm, rho=32):
    """ RMSE in Pixels """
    preds_px = preds_norm * rho
    labels_px = labels_norm * rho
    mse = F.mse_loss(preds_px, labels_px)
    return torch.sqrt(mse)

def run_validation(model, ValList, BatchSize, Rho, ModelType):
    model.eval()
    val_loss_sum = 0.0
    val_rmse_sum = 0.0
    criterion_sup = nn.L1Loss()
    criterion_unsup = nn.L1Loss()
    
    num_batches = 10
    
    with torch.no_grad():
        for _ in range(num_batches):
            batch = GenerateBatch(ValList, BatchSize, Rho, ModelType)
            
            if ModelType == "Sup":
                preds = model(batch["input"])
                loss = criterion_sup(preds, batch["label"])
                rmse = calculate_rmse(preds, batch["label"], Rho)
            else:
                warped_a, delta = model(batch["input"], batch["patch_a"], batch["corners"], Rho)
                loss = criterion_unsup(warped_a, batch["patch_b"])
                # For Unsup, we don't have GT delta labels easily available in this loop structure
                # unless we change GenerateBatch to always return them.
                # Just return 0 RMSE for Unsup or modify GenerateBatch.
                rmse = torch.tensor(0.0)

            val_loss_sum += loss.item()
            val_rmse_sum += rmse.item()
            
    return val_loss_sum / num_batches, val_rmse_sum / num_batches

def TrainOperation(TrainList, ValList, Args, LogsPath):
    model = UnifiedHomographyModel(model_type=Args.ModelType).to(device)
    optimizer = optim.Adam(model.parameters(), lr=Args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    criterion_sup = nn.L1Loss()
    criterion_unsup = nn.L1Loss()
    
    writer = SummaryWriter(LogsPath)

    print(f"[INFO] Training {Args.ModelType} Model on {len(TrainList)} images.")

    for Epoch in range(Args.epochs):
        model.train()
        train_loss_epoch = 0.0
        train_rmse_epoch = 0.0
        
        Iterations = max(1, int(len(TrainList) / Args.batch_size))
        loop = tqdm(range(Iterations), desc=f"Epoch {Epoch+1}/{Args.epochs}")

        for Iter in loop:
            batch = GenerateBatch(TrainList, Args.batch_size, Args.rho, Args.ModelType)
            optimizer.zero_grad()

            if Args.ModelType == "Sup":
                preds = model(batch["input"])
                loss = criterion_sup(preds, batch["label"])
                rmse = calculate_rmse(preds, batch["label"], Args.rho)
            else:
                warped_a, _ = model(batch["input"], batch["patch_a"], batch["corners"], Args.rho)
                loss = criterion_unsup(warped_a, batch["patch_b"])
                rmse = torch.tensor(0.0) # Placeholder

            loss.backward()
            optimizer.step()

            train_loss_epoch += loss.item()
            train_rmse_epoch += rmse.item()
            
            loop.set_postfix(loss=loss.item(), rmse=rmse.item())

        # Stats
        avg_train_loss = train_loss_epoch / Iterations
        avg_train_rmse = train_rmse_epoch / Iterations
        
        # Validation
        avg_val_loss, avg_val_rmse = run_validation(model, ValList, Args.batch_size, Args.rho, Args.ModelType)
        
        print(f"Epoch {Epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val RMSE: {avg_val_rmse:.2f}")
        
        # TensorBoard
        writer.add_scalars("Loss", {'Train': avg_train_loss, 'Validation': avg_val_loss}, Epoch)
        if Args.ModelType == "Sup":
            writer.add_scalars("Accuracy_RMSE_Pixels", {'Train': avg_train_rmse, 'Validation': avg_val_rmse}, Epoch)
        
        scheduler.step(avg_val_loss)
        
        # Save
        SaveName = os.path.join(Args.checkpoint_path, f"{Args.ModelType.lower()}_homography_net.pth")
        torch.save(model.regressor.state_dict(), SaveName)

    writer.close()
    print(f"[DONE] Saved to {SaveName}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", required=True, help="Folder with Train images")
    parser.add_argument("--val_path", required=True, help="Folder with Val images")
    parser.add_argument("--checkpoint_path", default="./Checkpoints/", help="Save path")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--rho", type=int, default=32)
    parser.add_argument("--ModelType", default="Sup", choices=["Sup", "Unsup"])
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint_path): os.makedirs(args.checkpoint_path)

    # Load Full File Paths
    train_list = glob.glob(os.path.join(args.train_path, "*"))
    val_list = glob.glob(os.path.join(args.val_path, "*"))

    if len(train_list) == 0:
        print(f"[ERROR] No images found in {args.train_path}")
        return

    TrainOperation(train_list, val_list, args, "Logs/")

if __name__ == "__main__":
    main()