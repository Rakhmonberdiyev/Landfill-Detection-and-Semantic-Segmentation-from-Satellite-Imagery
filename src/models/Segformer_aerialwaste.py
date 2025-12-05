import os, cv2, numpy as np
from PIL import Image
import torch, math
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import SegformerForSemanticSegmentation, SegformerFeatureExtractor
from torch.cuda.amp import autocast, GradScaler

images_root = "./images"
refined_masks_root = "./refined_masks"
test_masks_root = "./testing_masks"
train_landfill_dir = "./dataset_binary/train/landfill"

OUT_VIS = "./segformer_improved_vis"
OUT_TEST = "./segformer_improved_test"
os.makedirs(OUT_VIS, exist_ok=True)
os.makedirs(OUT_TEST, exist_ok=True)

IMG_SIZE = 256
BATCH_SIZE = 2
NUM_EPOCHS = 24
ENCODER_LR = 2e-5
HEAD_LR = 1e-4
WEIGHT_DECAY = 1e-2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 0

def load_mask(path, size=(IMG_SIZE,IMG_SIZE)):
    m = Image.open(path).convert("L").resize(size, Image.NEAREST)
    arr = (np.array(m) > 128).astype(np.float32)
    return arr

class TrainDataset:
    def __init__(self):
        self.samples = []
        for fname in sorted(os.listdir(train_landfill_dir)):
            if fname.lower().endswith((".png",".jpg",".jpeg")):
                img_path = os.path.join(images_root, fname)
                mask_path = os.path.join(refined_masks_root, fname.rsplit(".",1)[0] + "_mask.png")
                if os.path.exists(img_path) and os.path.exists(mask_path):
                    self.samples.append((img_path, mask_path))

        self.extractor = SegformerFeatureExtractor(do_resize=True, size=IMG_SIZE, do_normalize=True)

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]
        img = Image.open(img_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        mask = load_mask(mask_path)

        if np.random.rand() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = np.fliplr(mask).copy()

        if np.random.rand() < 0.5:
            arr = np.array(img).astype(np.float32)
            arr *= (0.9 + 0.2 * np.random.rand())
            arr = np.clip(arr, 0, 255).astype(np.uint8)
            img = Image.fromarray(arr)

        pixel_values = self.extractor(img, return_tensors="pt")["pixel_values"].squeeze(0)
        mask = torch.from_numpy(mask.copy()).unsqueeze(0)

        return pixel_values, mask, os.path.basename(img_path)

class TestDataset:
    def __init__(self):
        self.samples = []
        for fname in sorted(os.listdir(test_masks_root)):
            if fname.lower().endswith(".png"):
                mask_path = os.path.join(test_masks_root, fname)
                img_name = fname.replace("_mask","")
                img_path = os.path.join(images_root, img_name)
                if os.path.exists(img_path):
                    self.samples.append((img_path, mask_path))
        self.extractor = SegformerFeatureExtractor(do_resize=True, size=IMG_SIZE, do_normalize=True)

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]
        img = Image.open(img_path).convert("RGB").resize((IMG_SIZE,IMG_SIZE))
        mask = load_mask(mask_path)
        pixel_values = self.extractor(img, return_tensors="pt")["pixel_values"].squeeze(0)
        mask = torch.from_numpy(mask).unsqueeze(0)
        return pixel_values, mask, os.path.basename(img_path)

train_ds = TrainDataset()
test_ds = TestDataset()
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
test_loader  = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=NUM_WORKERS)

print("Train samples:", len(train_ds), "Test samples:", len(test_ds))

sum_fg, sum_total = 0.0, 0.0
for _, mpath in train_ds.samples:
    m = load_mask(mpath)
    sum_fg += m.sum()
    sum_total += m.size
pos_ratio = sum_fg / (sum_total + 1e-9)
print(f"Foreground pixel ratio: {pos_ratio:.6f}")
pos_weight_val = (1.0 - pos_ratio) / (pos_ratio + 1e-9)
pos_weight_val = float(max(1.0, min(pos_weight_val, 30.0)))
print("Using pos_weight for BCE:", pos_weight_val)

model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/segformer-b2-finetuned-ade-512-512",
    num_labels=1,
    ignore_mismatched_sizes=True
).to(DEVICE)

param_groups = []
backbone_names = ("encoder", "backbone")
for name, p in model.named_parameters():
    if any(k in name for k in ["decode_head", "classifier", "decode", "segformer_decoder"]):
        param_groups.append({"params": p, "lr": HEAD_LR, "weight_decay": WEIGHT_DECAY})
    else:
        param_groups.append({"params": p, "lr": ENCODER_LR, "weight_decay": WEIGHT_DECAY})

optimizer = optim.AdamW(param_groups, lr=ENCODER_LR, weight_decay=WEIGHT_DECAY)

scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS * len(train_loader))

bce_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight_val).to(DEVICE))
def dice_loss_fn(logits, targets):
    probs = torch.sigmoid(logits)
    inter = (probs * targets).sum(dim=(1,2,3)) * 2
    den = probs.sum(dim=(1,2,3)) + targets.sum(dim=(1,2,3)) + 1e-6
    return 1.0 - (inter / den).mean()

scaler = GradScaler()
best_iou = 0.0

global_step = 0
for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    for pix, mask, name in train_loader:
        global_step += 1
        pix = pix.to(DEVICE)
        mask = mask.to(DEVICE)

        optimizer.zero_grad()
        with autocast():
            outputs = model(pixel_values=pix)
            logits = outputs.logits
            logits_up = torch.nn.functional.interpolate(logits, size=(IMG_SIZE,IMG_SIZE), mode='bilinear', align_corners=False)

            loss_bce = bce_loss(logits_up, mask)
            loss_dice = dice_loss_fn(logits_up, mask)
            loss = 0.5 * loss_bce + 0.5 * loss_dice

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        running_loss += loss.item()

        if global_step % 200 == 0:
            with torch.no_grad():
                probs = torch.sigmoid(logits_up)
                preds = (probs > 0.5).float().cpu().numpy()
                im = (pix[0].permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
                gt = (mask[0,0].cpu().numpy() * 255).astype(np.uint8)
                pr = (preds[0,0] * 255).astype(np.uint8)
                overlay = cv2.addWeighted(im, 0.6, cv2.applyColorMap(pr, cv2.COLORMAP_JET), 0.4, 0)
                cv2.imwrite(os.path.join(OUT_VIS, f"ep{epoch+1}_step{global_step}_{name[0]}"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Avg loss: {running_loss/len(train_loader):.4f}")

    model.eval()
    total_iou, total_acc, cnt = 0.0, 0.0, 0
    with torch.no_grad():
        for pix, mask, name in test_loader:
            pix = pix.to(DEVICE)
            mask = mask.to(DEVICE)
            logits = model(pixel_values=pix).logits
            logits_up = torch.nn.functional.interpolate(logits, size=(IMG_SIZE,IMG_SIZE), mode='bilinear', align_corners=False)
            probs = torch.sigmoid(logits_up)
            pr = (probs > 0.5).float().cpu().numpy()[0,0]
            gt = mask.cpu().numpy()[0,0]
            inter = np.logical_and(pr==1, gt==1).sum()
            union = np.logical_or(pr==1, gt==1).sum()
            iou = inter / (union + 1e-6)
            acc = (pr==gt).sum() / pr.size
            total_iou += iou
            total_acc += acc
            cnt += 1
    mean_iou = total_iou / max(1, cnt)
    mean_acc = total_acc / max(1, cnt)
    print(f"Validation — Mean IoU: {mean_iou:.4f}, Pixel Acc: {mean_acc:.4f}")

    if mean_iou > best_iou:
        best_iou = mean_iou
        torch.save(model.state_dict(), "segformer_b2_improved_best.pth")
        print("Saved best model with IoU:", best_iou)

print("Training complete. Best IoU:", best_iou)
