#!/usr/bin/env python3
import os
import glob
import cv2
import numpy as np
import torch
import torch.nn as nn


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class HomographyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(2, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128 * 16 * 16, 1024),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(1024, 8),
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

        u = corners_A[..., 0]
        v = corners_A[..., 1]
        up = corners_B[..., 0]
        vp = corners_B[..., 1]

        A = torch.zeros((B, 8, 8), device=corners_A.device, dtype=corners_A.dtype)
        b = torch.zeros((B, 8, 1), device=corners_A.device, dtype=corners_A.dtype)

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

        H = torch.zeros((B, 3, 3), device=corners_A.device, dtype=corners_A.dtype)
        H[:, 0, 0] = hhat[:, 0, 0]; H[:, 0, 1] = hhat[:, 1, 0]; H[:, 0, 2] = hhat[:, 2, 0]
        H[:, 1, 0] = hhat[:, 3, 0]; H[:, 1, 1] = hhat[:, 4, 0]; H[:, 1, 2] = hhat[:, 5, 0]
        H[:, 2, 0] = hhat[:, 6, 0]; H[:, 2, 1] = hhat[:, 7, 0]; H[:, 2, 2] = 1.0
        return H


class UnsupervisedHomographyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.regressor = HomographyNet()
        self.dlt = TensorDLT()

    def forward(self, stacked, corners_a, rho):
        pred8 = self.regressor(stacked)  # (B,8)
        delta = (rho * torch.tanh(pred8)).view(-1, 4, 2)  # (B,4,2)
        H = self.dlt(corners_a, delta)  # (B,3,3)
        return H, delta


#Cylindrical warp
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
    mask = cv2.remap(np.ones((h, w), np.uint8) * 255, map_x, map_y, interpolation=cv2.INTER_NEAREST,
                     borderMode=cv2.BORDER_CONSTANT)
    mask = (mask > 0).astype(np.uint8) * 255
    return warped, mask


# Extract patch
def extract_patch_center(gray, cx, cy, crop):
    half = crop // 2
    x0 = int(round(cx)) - half
    y0 = int(round(cy)) - half
    return x0, y0, gray[y0:y0+crop, x0:x0+crop]


def valid_patch(x0, y0, H, W, crop, margin):
    return (x0 >= margin and y0 >= margin and x0 + crop < (W - margin) and y0 + crop < (H - margin))


#Homography
def compute_pairwise_H_from_matches(model, imgA, imgB,
                                    crop=128, rho=32, margin=32,
                                    max_matches=300, ransac_thresh=4.0):
    grayA = cv2.cvtColor(imgA, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(imgB, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(nfeatures=8000)
    kA, dA = orb.detectAndCompute(grayA, None)
    kB, dB = orb.detectAndCompute(grayB, None)
    if dA is None or dB is None or len(kA) < 20 or len(kB) < 20:
        raise RuntimeError("Not enough ORB keypoints.")

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(dA, dB)
    matches = sorted(matches, key=lambda m: m.distance)[:max_matches]
    if len(matches) < 20:
        raise RuntimeError("Not enough ORB matches.")

    Hh, Ww = grayA.shape[:2]
    H2, W2 = grayB.shape[:2]

    corners_patch = np.array([[0, 0],
                              [crop-1, 0],
                              [crop-1, crop-1],
                              [0, crop-1]], dtype=np.float32)

    stacked_list = []
    corners_list = []
    meta = []  # (x0A, y0A, x0B, y0B)

    for m in matches:
        xa, ya = kA[m.queryIdx].pt
        xb, yb = kB[m.trainIdx].pt

        x0A, y0A, PA = extract_patch_center(grayA, xa, ya, crop)
        x0B, y0B, PB = extract_patch_center(grayB, xb, yb, crop)

        if not valid_patch(x0A, y0A, Hh, Ww, crop, margin):
            continue
        if not valid_patch(x0B, y0B, H2, W2, crop, margin):
            continue
        if PA.shape != (crop, crop) or PB.shape != (crop, crop):
            continue
        if PA.std() < 5 or PB.std() < 5:
            continue

        PA = PA.astype(np.float32) / 255.0
        PB = PB.astype(np.float32) / 255.0

        stacked_list.append(np.stack([PA, PB], axis=0))
        corners_list.append(corners_patch)
        meta.append((x0A, y0A, x0B, y0B))

    if len(stacked_list) < 12:
        raise RuntimeError("Not enough valid patch pairs from matches.")

    stacked = torch.from_numpy(np.stack(stacked_list, axis=0)).float().to(device)
    corners = torch.from_numpy(np.stack(corners_list, axis=0)).float().to(device)

    model.eval()
    with torch.no_grad():
        _, delta = model(stacked, corners, rho=rho)  # (B,4,2)
    delta = delta.cpu().numpy()

    # Build image-space correspondences
    ptsA, ptsB = [], []
    for i, (x0A, y0A, x0B, y0B) in enumerate(meta):
        cA = corners_patch + np.array([x0A, y0A], dtype=np.float32)
        cB = (corners_patch + delta[i]) + np.array([x0B, y0B], dtype=np.float32)
        ptsA.append(cA)
        ptsB.append(cB)

    ptsA = np.vstack(ptsA).astype(np.float32)
    ptsB = np.vstack(ptsB).astype(np.float32)

    H_AB, inl = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, ransacReprojThreshold=ransac_thresh)
    if H_AB is None:
        raise RuntimeError("findHomography failed.")
    H_AB = (H_AB / H_AB[2, 2]).astype(np.float32)
    return H_AB


#Blending
def feather_weight(mask):
    mask = (mask > 0).astype(np.uint8)
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    if dist.max() < 1e-6:
        return dist.astype(np.float32)
    w = dist / (dist.max() + 1e-6)
    w = cv2.GaussianBlur(w, (51, 51), 0)
    w = w / (w.max() + 1e-6)
    return w.astype(np.float32)


def warp_with_mask(img, H, out_w, out_h):
    warped = cv2.warpPerspective(img, H, (out_w, out_h), flags=cv2.INTER_LINEAR)
    base_mask = np.ones((img.shape[0], img.shape[1]), np.uint8) * 255
    wmask = cv2.warpPerspective(base_mask, H, (out_w, out_h), flags=cv2.INTER_NEAREST)
    return warped, wmask


# Stich folder
def stitch_folder(folder_path, model_path, out_path="mypano.png",
                  crop=128, rho=32, margin=32,
                  max_matches=300, ransac_thresh=4.0,
                  cyl_focal_scale=1.0):

    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    paths = []
    for e in exts:
        paths.extend(glob.glob(os.path.join(folder_path, e)))
    paths = sorted(paths)
    if len(paths) < 2:
        raise RuntimeError("Need at least 2 images in the folder.")

    imgs = []
    for p in paths:
        im = cv2.imread(p, cv2.IMREAD_COLOR)
        if im is not None:
            imgs.append(im)
    if len(imgs) < 2:
        raise RuntimeError("Could not read enough images.")

    # Cylindrical warp for all images
    h0, w0 = imgs[0].shape[:2]
    f = float(cyl_focal_scale * w0)

    cyl_imgs = []
    for im in imgs:
        wim, _ = cylindrical_warp(im, f=f)
        cyl_imgs.append(wim)

    # Load model
    model = UnsupervisedHomographyModel().to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # Big canvas
    h0, w0 = cyl_imgs[0].shape[:2]
    canvas_h = int(h0 * 2.0)
    canvas_w = int(w0 * 3.0)

    T_init = np.array([[1, 0, canvas_w // 3],
                       [0, 1, canvas_h // 2],
                       [0, 0, 1]], dtype=np.float32)

    acc = np.zeros((canvas_h, canvas_w, 3), dtype=np.float32)
    accw = np.zeros((canvas_h, canvas_w), dtype=np.float32)

    w0_img, m0 = warp_with_mask(cyl_imgs[0], T_init, canvas_w, canvas_h)
    w = feather_weight(m0)
    acc += w0_img.astype(np.float32) * w[..., None]
    accw += w

    C_prev = T_init.copy()

    for i in range(1, len(cyl_imgs)):
        print(f"[INFO] Stitching {i}/{len(cyl_imgs)-1}")

        H_prev_to_i = compute_pairwise_H_from_matches(
            model, cyl_imgs[i-1], cyl_imgs[i],
            crop=crop, rho=rho, margin=margin,
            max_matches=max_matches, ransac_thresh=ransac_thresh
        )

        # map i -> canvas
        C_i = C_prev @ np.linalg.inv(H_prev_to_i)
        C_i = (C_i / C_i[2, 2]).astype(np.float32)

        wimg, m = warp_with_mask(cyl_imgs[i], C_i, canvas_w, canvas_h)
        w = feather_weight(m)

        acc += wimg.astype(np.float32) * w[..., None]
        accw += w
        C_prev = C_i

    pano = (acc / np.clip(accw[..., None], 1e-6, None)).clip(0, 255).astype(np.uint8)

    # Crop
    gray = cv2.cvtColor(pano, cv2.COLOR_BGR2GRAY)
    ys, xs = np.where(gray > 0)
    if len(xs) > 0 and len(ys) > 0:
        x0, x1 = xs.min(), xs.max()
        y0, y1 = ys.min(), ys.max()
        pano = pano[y0:y1+1, x0:x1+1]

    cv2.imwrite(out_path, pano)
    print(f"[DONE] Saved panorama: {out_path}")


if __name__ == "__main__":
    FOLDER = "/home/alien/YourDirectoryID_p1/Phase1/Data/Test/TestSet3"
    MODEL  = "/home/alien/YourDirectoryID_p1/Phase2/Code/unsupervised_homography_net.pth"
    OUT    = "mypano.png"

    stitch_folder(
        folder_path=FOLDER,
        model_path=MODEL,
        out_path=OUT,
        crop=128,
        rho=32,
        margin=32,
        max_matches=300,
        ransac_thresh=4.0,
        cyl_focal_scale=1.0
    )
