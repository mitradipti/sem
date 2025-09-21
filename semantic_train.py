from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import os
import torch
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation

class SegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir, feature_extractor):
        self.images = sorted([os.path.join(images_dir, f) for f in os.listdir(images_dir) if f.endswith('.jpg')])
        self.masks = sorted([os.path.join(masks_dir, f) for f in os.listdir(masks_dir) if f.endswith('.png')])
        self.feature_extractor = feature_extractor

    def __len__(self):
        return len(self.images)

    
    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert("RGB")
        mask = Image.open(self.masks[idx])

    # Resize both image and mask to a fixed size (e.g., 512x512)
        target_size = (512, 512)
        image = image.resize(target_size, Image.BILINEAR)
        mask = mask.resize(target_size, Image.NEAREST)  

        mask = np.array(mask)  # mask contains class indices 0,1,2,3,4,5,6,7,8,9

        inputs = self.feature_extractor(images=image, return_tensors="pt")
    # Remove batch dimension
        inputs = {k: v.squeeze() for k, v in inputs.items()}

    # Resize mask to model output size (e.g., 160x160)
        output_size = (inputs['pixel_values'].shape[1] // 4, inputs['pixel_values'].shape[2] // 4)
        mask = Image.fromarray(mask).resize(output_size, Image.NEAREST)
        mask = np.array(mask)

        return inputs, torch.tensor(mask, dtype=torch.long)

# dataset Paths
images_dir = r"C:\Users\oishi\Documents\pm2.5\semantic_seg\dataset\original_data" # Put original_data path 
masks_dir = r"C:\Users\oishi\Documents\pm2.5\semantic_seg\dataset\semantic_annotations_mask" # Put semantic_annotations_mask path
num_classes = 10  

# Load feature extractor and model
feature_extractor = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b5-finetuned-ade-640-640")
model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/segformer-b5-finetuned-ade-640-640",
    num_labels=num_classes,
    ignore_mismatched_sizes=True
)

# Dataset and DataLoader
dataset = SegmentationDataset(images_dir, masks_dir, feature_extractor)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
loss_fn = torch.nn.CrossEntropyLoss()

# Training loop (simple version)
model.train()
for epoch in range(10):  # Set your number of epochs
    for batch in dataloader:
        inputs, masks = batch
        for k in inputs:
            inputs[k] = inputs[k].to(device)
        masks = masks.to(device)
        outputs = model(**inputs)
        loss = loss_fn(outputs.logits, masks)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    print(f"Epoch {epoch+1}: Loss {loss.item():.4f}")

# Save the fine-tuned model
model.save_pretrained(r"C:\Users\oishi\Documents\pm2.5\segformer_finetuned_trained") # Adjust the path (here segformer_finetuned_trained is a file name. After training, it will be saved as segformer_finetuned_trained.pth file)

feature_extractor.save_pretrained(r"C:\Users\oishi\Documents\pm2.5\segformer_finetuned_trained") # Adjust the path


