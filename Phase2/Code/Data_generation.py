import os
import glob 
import cv2
import numpy as np 


TRAIN_DIR = "/home/alien/YourDirectoryID_p1/Phase2/Data/Train"
OUTPUT_DIR = "/home/alien/YourDirectoryID_p1/Phase2/Data/Generated_Train"

MP, NP = 128, 128
rho = 32
num_samples = 20
use_gray = True
allow_translation = True

os.makedirs(OUTPUT_DIR, exist_ok=True)

img_paths = sorted(glob.glob(os.path.join(TRAIN_DIR, "**", "*.jpg"), recursive= True))

if len(img_paths) == 0:
    raise RuntimeError(f"no images found in {TRAIN_DIR}")

def random_patch(H,W, MP, NP, rho):
    x_min = rho
    y_min = rho 
    x_max = W-NP-rho
    y_max = H-MP-rho

    if x_max <= x_min or y_max <= y_min:
        return None

    x = np.random.randint(x_min, x_max + 1)
    y = np.random.randint(y_min, y_max + 1)

def build_CA(x, y, NP, MP):
    """
    Corner ordering MUST be consistent.
    We'll use:
      top-left, top-right, bottom-right, bottom-left
    """
    CA = np.array([
        [x,      y     ],
        [x+NP-1, y     ],
        [x+NP-1, y+MP-1],
        [x,      y+MP-1],
    ], dtype=np.float32)
    return CA


def perturb_corners(CA, rho, allow_translation):
    pert = np.random.randint(-rho, rho+1, size = (4,2)).astype(np.float32)

    if allow_translation:
        tx = np.random.randint(-rho, rho+1)
        ty = np.random.randint(-rho, rho+1)
        trans = np.array([tx,ty], dtype = np.float32)
    else:
        trans = np.zeros((2,), dtype= np.float32)
    
    CB = CA + pert +  trans
    return CB   

def crop_patch(img, x, y, NP, MP):
    return img[y:y+MP, x:x+NP].copy()

saved = 0
tries = 0
max_tries = num_samples*20

while saved < num_samples and tries < max_tries:
    tries += 1
    img_path = img_paths[np.random.randint(0, len(img_paths))]
    IA = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if IA is None:
        continue

    H, W = IA.shape[:2]

    tl = random_patch(H,W, MP, NP, rho)
    if tl is None:
        continue
    x, y = tl

    if use_gray:
        IA_proc = cv2.cvtColor(IA, cv2.COLOR_BGR2GRAY)
    else:
        IA_proc = IA
    
    CA = build_CA(x, y, NP, MP)
    CB = perturb_corners(CA, rho, allow_translation)

    if np.any(CB[:, 0] < 0) or np.any(CB[:, 0] >= W) or np.any(CB[:, 1] < 0) or np.any(CB[:, 1] >= H):
        continue

    H_AB = cv2.getPerspectiveTransform(CA, CB)
    if H_AB is None:
        continue

    try:
        H_BA = np.linalg.inv(H_AB)
    except np.linalg.LinAlgError:
        continue

    IB = cv2.warpPerspective(IA_proc, H_BA, (W, H), flags=cv2.INTER_LINEAR)

    PA = crop_patch(IA_proc, x, y, NP, MP)
    PB = crop_patch(IB,      x, y, NP, MP)

    if PA.shape[0] != MP or PA.shape[1] != NP or PB.shape[0] != MP or PB.shape[1] != NP:
        continue


    if use_gray:
        inp = np.stack([PA, PB], axis=-1).astype(np.float32) / 255.0
    else:
        PA_f = PA.astype(np.float32) / 255.0
        PB_f = PB.astype(np.float32) / 255.0
        inp = np.concatenate([PA_f, PB_f], axis=-1)

    H4Pt = (CB - CA).reshape(-1).astype(np.float32)  

    # Save
    out_path = os.path.join(OUTPUT_DIR, f"sample_{saved:06d}.npz")
    np.savez_compressed(out_path, input=inp, label=H4Pt)

    saved += 1

print(f"Done. Saved {saved} samples to: {OUTPUT_DIR}")
print(f"Total tries: {tries}")

