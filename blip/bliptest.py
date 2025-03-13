from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch

# Set device
device = "mps" if torch.backends.mps.is_available() else "cpu"
print("Using device:", device)

device = "cpu"
print("Using device:", device)

# Load model and processor, can also use blip 2
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)  # ✅ Move model to device

# Load image
image_path = "../in/graveltest.jpg"
raw_image = Image.open(image_path).convert('RGB')

# Prepare inputs and move to device
inputs = processor(images=raw_image, return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}  # ✅ Move each tensor to device

# Generate caption
with torch.no_grad():
    generated_ids = model.generate(**inputs, max_new_tokens=50)
    caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

print("Caption:", caption)
