from sentence_transformers import SentenceTransformer
from torch.utils.data import Dataset
from PIL import Image
import os
import torch
import numpy as np
import torchvision.transforms as transforms
import torch.nn as nn
from torch.utils.data import DataLoader


def prepare_data(text_file, image_dir):

    class RoomDatasetCond(Dataset):
        def __init__(self, text_file, image_dir, transform=None):
            self.image_dir = image_dir
            self.transform = transform
            self.data = []
            self.model = SentenceTransformer("nomic-ai/nomic-embed-text-v1", trust_remote_code=True)
            if not hasattr(self.model, "output_dim"):
                self.model.output_dim = 768  # adjust as needed

            # Read the text file to link descriptions with images
            with open(text_file, 'r') as f:
                description = ""
                for i, line in enumerate(f):
                    line = line.strip()  # Remove leading/trailing whitespace

                    # If '#####' is found, we know that the line contains an image reference
                    if '#####' in line:
                        # Split only on the first occurrence of '#####'
                        parts = line.split('#####', 1)
                        description = parts[0].strip()  # Get the description part
                        image_ref = parts[1].strip()  # Get the image reference part

                        # Remove the trailing '######' from the image reference
                        if image_ref.endswith('######'):
                            image_ref = image_ref[:-6].strip()

                        # Process the image reference
                        parts = image_ref.strip().split("\\")
                        room_type = parts[0]  # Bathroom/Kitchen/...
                        image_number = parts[1].replace('.jpg', '')  # Remove the .jpg extension

                        # Construct the full image path
                        image_filename = f"{image_number}.jpg"
                        image_path = os.path.join(self.image_dir, room_type, image_filename)

                        # Check if the image exists
                        if not os.path.exists(image_path):
                            print(f"Warning: Image '{image_filename}' not found. Skipping description.","The Room Type was",room_type)
                            continue  # Skip this entry if image is not found

                        # Add the description and image path to the dataset
                        text_embedding = self.get_text_embedding(description)  # Get text embedding
                        self.data.append((text_embedding, image_path))

            # If the last description does not end with a proper '#####', ensure it's handled
            if not line.endswith('#####'):
                print("Warning: Last description is not followed by '#####', but it will be linked correctly.")

        def get_text_embedding(self, text):
            with torch.no_grad():
                outputs = self.model.encode(text)
            return outputs

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            text_embedding, image_path = self.data[idx]

            image = Image.open(image_path).convert('RGB')
            if self.transform:
                image = self.transform(image)

            return text_embedding, image
    # Define image transformations
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
    dataset = RoomDatasetCond(text_file=text_file, image_dir=image_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    return dataloader