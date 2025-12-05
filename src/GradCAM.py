import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

data_root = "./dataset_binary"
generated_masks_dir = "./generated_masks"
os.makedirs(generated_masks_dir, exist_ok=True)

batch_size = 16
num_epochs = 20
learning_rate = 1e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

train_dataset = datasets.ImageFolder(os.path.join(data_root, "train"), transform_train)
test_dataset  = datasets.ImageFolder(os.path.join(data_root, "test"),  transform_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

class_names = train_dataset.classes
print("Classes:", class_names)

model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
model.fc = nn.Linear(model.fc.in_features, 2) 
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss = {total_loss/len(train_loader):.4f}")

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()

        self.gradients = None
        self.activations = None

        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, input_tensor):
        self.model.zero_grad()

        output = self.model(input_tensor)
        class_idx = output.argmax(dim=1)
        score = output[:, class_idx]

        score.backward(retain_graph=True)

        gradients = self.gradients
        activations = self.activations

        alpha = gradients.mean(dim=[2, 3], keepdim=True)
        heatmap = (alpha * activations).sum(dim=1).squeeze()
        heatmap = torch.relu(heatmap)

        heatmap -= heatmap.min()
        heatmap /= heatmap.max()
        return heatmap.detach().cpu().numpy()

target_layer = model.layer4[-1]
gradcam = GradCAM(model, target_layer)

landfill_class_idx = class_names.index("landfill")

landfill_dataset = datasets.ImageFolder(
    os.path.join(data_root, "train"),
    transform_test
)

for i, (img_tensor, label) in enumerate(landfill_dataset):

    if label != landfill_class_idx:
        continue 

    img_path = landfill_dataset.samples[i][0]
    filename = os.path.basename(img_path)

    input_tensor = img_tensor.unsqueeze(0).to(device)

    heatmap = gradcam.generate(input_tensor)
    heatmap = cv2.resize(heatmap, (224, 224))

    mask = (heatmap * 255).astype(np.uint8)
    mask_path = os.path.join(generated_masks_dir, filename.replace(".png", "_mask.png"))
    cv2.imwrite(mask_path, mask)

    original_img = Image.open(img_path).convert("RGB")
    original_img = original_img.resize((224, 224))
    original_np = np.array(original_img)

    heatmap_color = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    overlay = (0.5 * original_np + 0.5 * heatmap_color).astype(np.uint8)

    overlay_path = os.path.join(generated_masks_dir, filename.replace(".png", "_overlay.png"))
    Image.fromarray(overlay).save(overlay_path)

    print(f"Saved: {mask_path}")
    print(f"Saved: {overlay_path}")

print("\nDone")
