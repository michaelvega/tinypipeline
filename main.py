from datasets import load_dataset
import os
from tqdm import tqdm

# Load the dataset in streaming mode to avoid full download
dataset = load_dataset("laion/laion400m", split="train", streaming=True)

# Folder to store URLs file
os.makedirs("gravel_images", exist_ok=True)

# Simple keyword filter
keywords = ['gravel']
exclude_keywords = ['bag', 'packaging', 'store', 'bought', 'sack']

# File to store the collected URLs
url_file_path = "gravel_images/gravel_image_urls.txt"

# Open file to write URLs
with open(url_file_path, 'w') as url_file:
    print("Starting streaming and filtering...")

    count = 0
    max_urls = 200  # Change this number to collect more/less

    for item in tqdm(dataset):
        try:
            # Check if item is valid
            if not item or 'caption' not in item or 'url' not in item:
                continue  # Skip incomplete or broken entries

            caption = item['caption'].lower()
            url = item['url']

            # Check if caption contains desired keywords and exclude bad ones
            if any(k in caption for k in keywords) and not any(bad in caption for bad in exclude_keywords):
                print(f"[{count}] Found gravel image: {caption}")

                # Save URL to file
                url_file.write(url + '\n')
                count += 1

            if count >= max_urls:
                break

        except Exception as e:
            print(f"Error processing item: {e}")

print(f"Finished collecting URLs! Total collected: {count}")
print(f"Saved to {url_file_path}")
