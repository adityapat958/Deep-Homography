#!/usr/bin/env python3
"""
RBE/CS Fall 2022: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project 1: MyAutoPano: Phase 2 (Unsupervised Deep Homography)

"""
import os
import glob
import numpy as np
import cv2
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class HomographyDataset(Dataset):
    def __init__(self, root_dir, crop_size=128, boundary_limit=32, transformation=None):
        super().__init__()
        self.root_dir = root_dir
        self.crop_size = crop_size
        self.transformation = transformation
        self.image_path = glob.glob(os.path.join(root_dir, "*"))
        self.boundary_limit = boundary_limit

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, idx):
        img_path = self.image_path[idx]
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        # Safety check
        if (image is None or
            image.shape[1] < self.crop_size + 2 * self.boundary_limit or
            image.shape[0] < self.crop_size + 2 * self.boundary_limit):
            return self.__getitem__(np.random.randint(0, len(self)))

        pa, pb, gt_delta = self.get_random_crop_and_pertube(image, self.crop_size, self.crop_size)

        if pa is None:
            return self.__getitem__(np.random.randint(0, len(self)))

        # Normalize patches to [0,1]
        pa = pa.astype(np.float32) / 255.0
        pb = pb.astype(np.float32) / 255.0

        # Stack (2,H,W) for the regressor input
        stacked = np.stack([pa, pb], axis=0)

        W = self.crop_size
        H = self.crop_size
        corners = np.array([[0, 0],
                            [W - 1, 0],
                            [W - 1, H - 1],
                            [0, H - 1]], dtype=np.float32)

        label = gt_delta.reshape(-1).astype(np.float32) 

        sample = {
            "image": torch.from_numpy(stacked),             
            "patch_a": torch.from_numpy(pa)[None, ...],    
            "patch_b": torch.from_numpy(pb)[None, ...],   
            "corners": torch.from_numpy(corners),           
            "label": torch.from_numpy(label)             
        }

        if self.transformation:
            sample = self.transformation(sample)

        return sample

    def get_random_crop_and_pertube(self, image, crop_height=128, crop_width=128):
        boundary_limit = self.boundary_limit

        max_x = image.shape[1] - crop_width - boundary_limit
        max_y = image.shape[0] - crop_height - boundary_limit
        if max_x <= boundary_limit or max_y <= boundary_limit:
            return None, None, None

        x = np.random.randint(boundary_limit, max_x)
        y = np.random.randint(boundary_limit, max_y)

        crop_image = image[y:y + crop_height, x:x + crop_width]

        corners_img = np.array([[x, y],
                                [x + crop_width - 1, y],
                                [x + crop_width - 1, y + crop_height - 1],
                                [x, y + crop_height - 1]], dtype=np.float32)

        pertube_range = boundary_limit - 1
        pert = np.random.randint(-pertube_range, pertube_range + 1, (4, 2)).astype(np.float32)
        corners_img_pert = corners_img + pert

        H = cv2.getPerspectiveTransform(corners_img, corners_img_pert)

        warped_full = cv2.warpPerspective(
            image,
            H,
            (image.shape[1], image.shape[0]),
            flags=cv2.WARP_INVERSE_MAP
        )

        crop_pert = warped_full[y:y + crop_height, x:x + crop_width]

        gt_delta = pert.copy() 

        return crop_image, crop_pert, gt_delta


class HomographyNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(128 * 16 * 16, 1024),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 8)
        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.contiguous().view(x.size(0), -1)
        out = self.fc(out)
        return out


class TensorDLT(nn.Module):
    def forward(self, corners_A, delta):

        B = corners_A.shape[0]
        corners_B = corners_A + delta  

        u = corners_A[..., 0]
        v = corners_A[..., 1]
        up = corners_B[..., 0]
        vp = corners_B[..., 1]

        A = torch.zeros((B, 8, 8), device=corners_A.device, dtype=corners_A.dtype)
        b = torch.zeros((B, 8, 1), device=corners_A.device, dtype=corners_A.dtype)

        for i in range(4):
       
            A[:, 2 * i, 3] = -u[:, i]
            A[:, 2 * i, 4] = -v[:, i]
            A[:, 2 * i, 5] = -1.0
            A[:, 2 * i, 6] = vp[:, i] * u[:, i]
            A[:, 2 * i, 7] = vp[:, i] * v[:, i]
            b[:, 2 * i, 0] = -vp[:, i]

            A[:, 2 * i + 1, 0] = u[:, i]
            A[:, 2 * i + 1, 1] = v[:, i]
            A[:, 2 * i + 1, 2] = 1.0
            A[:, 2 * i + 1, 6] = -up[:, i] * u[:, i]
            A[:, 2 * i + 1, 7] = -up[:, i] * v[:, i]
            b[:, 2 * i + 1, 0] = up[:, i]

        hhat = torch.linalg.lstsq(A, b).solution 

        H = torch.zeros((B, 3, 3), device=corners_A.device, dtype=corners_A.dtype)
        H[:, 0, 0] = hhat[:, 0, 0]
        H[:, 0, 1] = hhat[:, 1, 0]
        H[:, 0, 2] = hhat[:, 2, 0]
        H[:, 1, 0] = hhat[:, 3, 0]
        H[:, 1, 1] = hhat[:, 4, 0]
        H[:, 1, 2] = hhat[:, 5, 0]
        H[:, 2, 0] = hhat[:, 6, 0]
        H[:, 2, 1] = hhat[:, 7, 0]
        H[:, 2, 2] = 1.0
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
    pB = torch.stack([xs, ys, ones], dim=-1)            
    pB = pB.view(1, out_h * out_w, 3).repeat(B, 1, 1)  


    Hinv = torch.linalg.inv(H)
    pA = (Hinv @ pB.transpose(1, 2)).transpose(1, 2)  

    den = pA[..., 2].clamp(min=1e-8)
    xA = pA[..., 0] / den
    yA = pA[..., 1] / den

 
    xA_norm = (2.0 * xA / (Wa - 1)) - 1.0
    yA_norm = (2.0 * yA / (Ha - 1)) - 1.0
    grid = torch.stack([xA_norm, yA_norm], dim=-1)     
    grid = grid.view(B, out_h, out_w, 2)

    warped = F.grid_sample(
        patch_a,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True
    )
    return warped



class UnsupervisedHomographyModel(nn.Module):
    def __init__(self, regressor):
        super().__init__()
        self.regressor = regressor
        self.dlt = TensorDLT()

    def forward(self, stacked, patch_a, corners_a, rho):
        pred8 = self.regressor(stacked)                    
        delta = (rho * torch.tanh(pred8)).view(-1, 4, 2)  
        H = self.dlt(corners_a, delta)                  
        warped_a = warp_patch(patch_a, H, 128, 128)    
        return warped_a, delta, H



class UnsupervisedTrainer:
    def __init__(self, model, train_loader, val_loader, lr=1e-4, epochs=100, boundary_limit=32):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.lr = lr
        self.epochs = epochs
        self.rho = boundary_limit - 1
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def train(self):
        print("[INFO] Starting UNSUPERVISED training...")
        for epoch in range(self.epochs):
            self.model.train()
            running = 0.0
            loop = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.epochs}")

            for batch in loop:
                stacked = batch["image"].float().to(device)       
                patch_a = batch["patch_a"].float().to(device)   
                patch_b = batch["patch_b"].float().to(device)    
                corners = batch["corners"].float().to(device)    

                warped_a, _, _ = self.model(stacked, patch_a, corners, self.rho)
                loss = torch.mean(torch.abs(warped_a - patch_b))  # photometric L1

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()

                running += loss.item()
                loop.set_postfix(loss=float(loss.item()))

            val_loss = self.validate()
            print(f"Epoch [{epoch+1}/{self.epochs}] TrainLoss {running/len(self.train_loader):.6f}  ValLoss {val_loss:.6f}")

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        running = 0.0
        for batch in self.val_loader:
            stacked = batch["image"].float().to(device)
            patch_a = batch["patch_a"].float().to(device)
            patch_b = batch["patch_b"].float().to(device)
            corners = batch["corners"].float().to(device)

            warped_a, _, _ = self.model(stacked, patch_a, corners, self.rho)
            loss = torch.mean(torch.abs(warped_a - patch_b))
            running += loss.item()
        return running / len(self.val_loader)

    def save(self, path):
        torch.save(self.model.state_dict(), path)
        print(f"[INFO] Model saved to: {path}")



def main():
    train_dir = "/home/alien/YourDirectoryID_p1/Phase2/Data/Train"
    val_dir = "/home/alien/YourDirectoryID_p1/Phase2/Data/Val"

    dataset = HomographyDataset(train_dir, crop_size=128, boundary_limit=32, transformation=None)
    val_dataset = HomographyDataset(val_dir, crop_size=128, boundary_limit=32, transformation=None)

    train_loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)

    regressor = HomographyNet()
    model = UnsupervisedHomographyModel(regressor)

    trainer = UnsupervisedTrainer(model, train_loader, val_loader, lr=1e-4, epochs=100, boundary_limit=32)
    trainer.train()
    trainer.save("unsupervised_homography_net.pth")


if __name__ == "__main__":
    main()
