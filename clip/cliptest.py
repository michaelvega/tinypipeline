import torch

device = "mps" if torch.backends.mps.is_available() else "cpu"
print("Using device:", device)

import clip
import torch
from PIL import Image

# Set device
device = "mps" if torch.backends.mps.is_available() else "cpu"
print("Using device:", device)

# Load model
# worst is ViT-B/32, then ViT-L/14, then ViT-H/14, then ViT-G/14
model, preprocess = clip.load("ViT-B/32", device=device)

# Load and preprocess image
image_path = "../in/treee.jpeg"
image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

# Text prompts
prompts = [
    "a photo of gravel",
    "a photo of a gravel road",
    "a close-up of gravel on the ground",
    "a rocky gravel pathway",
    "gray gravel stones covering the ground",
    "text",
    "black",
    "a photo of a tree"
]
text = clip.tokenize(prompts).to(device)

# Similarity calculation
with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)

    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    similarity = (image_features @ text_features.T).squeeze(0).cpu().numpy()

# Output
for p, s in zip(prompts, similarity):
    print(f"Similarity to '{p}': {s:.3f}")