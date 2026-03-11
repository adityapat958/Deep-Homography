#!/usr/bin/env python
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

# --- CONFIGURATION ---
LOG_DIR = "Logs/Architecture_Diagram_With_Dropout" 

# ==========================================
# MODEL DEFINITION (With Dropout 0.1)
# ==========================================
class HomographyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(2,64,3,padding=1,bias=False), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.Conv2d(64,64,3,padding=1,bias=False), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.MaxPool2d(2,2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64,64,3,padding=1,bias=False), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.Conv2d(64,64,3,padding=1,bias=False), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.MaxPool2d(2,2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64,128,3,padding=1,bias=False), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.Conv2d(128,128,3,padding=1,bias=False), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.MaxPool2d(2,2)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(128,128,3,padding=1,bias=False), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.Conv2d(128,128,3,padding=1,bias=False), nn.BatchNorm2d(128), nn.ReLU(True)
        )
        # --- DROPOUT INCLUDED HERE ---
        self.fc = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(128*16*16, 1024), 
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

def main():
    # 1. Initialize Model
    model = HomographyNet()
    
    # 2. Create Dummy Input
    dummy_input = torch.randn(1, 2, 128, 128)

    # 3. Write Graph
    print(f"[INFO] Generating graph log in {LOG_DIR}...")
    writer = SummaryWriter(LOG_DIR)
    writer.add_graph(model, dummy_input)
    writer.close()
    print("[DONE] Graph saved.")
    print(f"Run: tensorboard --logdir {LOG_DIR}")

if __name__ == "__main__":
    main()