import argparse
import os
import torch
import h5py
import matplotlib.pyplot as plt
import numpy as np

from configs.action_dicts import ACTION_DICTS
from utils.robot_dataset import RobotDataset
from utils.vit_clip_model import ViTCLIPModel
from utils.losses import contrastive_loss, cosine_similarity_manual

from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torch.amp import autocast, GradScaler
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image



def train_embed(args):
    # 1) Select correct action dictionary
    if args.task not in ACTION_DICTS:
        raise ValueError(f"Unknown task: {args.task}")
    action_description = ACTION_DICTS[args.task]

    description_to_index = {v: k for k, v in action_description.items()}

    def extract_embedding(model, folder_path, action, device):
        model.eval()

        # --- 1. Collect and transform all images from the folder ---
        transform = transforms.Compose([
            transforms.Resize((224, 224)),  # or whatever your model expects
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
        ])

        # Collect all image files (PNG, JPG, etc.)
        image_files = sorted(
            f for f in os.listdir(folder_path)
            if f.lower().endswith(".png") or f.lower().endswith(".jpg")
        )
        if not image_files:
            # If no images found, return an empty or zero tensor
            return torch.zeros(model.fc1.in_features, device=device)

        # Load & transform each image into a list of tensors
        image_tensors = []
        for img_file in image_files:
            img_path = os.path.join(folder_path, img_file)
            image = Image.open(img_path).convert("RGB")
            image_tensors.append(transform(image))
        
        # Stack => shape [seq_len, 3, H, W]
        images_tensor = torch.stack(image_tensors, dim=0)

        # Add a batch dimension => shape [1, seq_len, 3, H, W]
        images_tensor = images_tensor.unsqueeze(0).to(device)

        # --- 2. Pass the batch of frames + single action to the model ---
        with torch.no_grad():
            # model.get_embedding => shape [1, emb_dim], after averaging frames in encode_images
            final_embedding = model.get_embedding(images_tensor, [action])
            # For a single sample, remove the batch dimension
            final_embedding = final_embedding.squeeze(0)  # shape [emb_dim]
        
        return final_embedding.cpu()



    transform_img = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std =[0.229, 0.224, 0.225],
        )
    ])
    actions_file = os.path.join(args.path, "actions.txt")
    success_file = os.path.join(args.path, "success_rate.txt")
    image_folder = os.path.join(args.path, "demo")

    dataset = RobotDataset(
        actions_file,
        success_file,
        image_folder,
        action_description,   # pass in the dictionary
        transform_img
    )
    
    # Subset if desired
    subset_indices = range(200)
    train_subset = Subset(dataset, subset_indices)

    dataloader = DataLoader(
        train_subset,
        batch_size=8,
        shuffle=True,
        num_workers=8,
        pin_memory=True
    )

        # Subset if desired
    subset_indices = range(200)
    train_subset = Subset(dataset, subset_indices)

    dataloader = DataLoader(
        train_subset,
        batch_size=8,
        shuffle=True,
        num_workers=8,
        pin_memory=True
    )

    # 3) Model, optimizer, etc.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ViTCLIPModel(checkpoint_gradient=args.checkpoint_gradient).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    scaler = GradScaler()

    print('training')

    # 4) Training loop
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for batch in dataloader:
            actions_text = batch["action"]
            success = batch["success"].float().to(device)
            images = batch["images"].to(device)

            action_labels = torch.tensor(
                [description_to_index[a] for a in actions_text],
                dtype=torch.long,
                device=device
            )

            optimizer.zero_grad()

            with autocast(dtype=torch.float16, device_type='cuda'):
                # classification
                logits = model(images, actions_text)
                loss_class = criterion(logits, success)

                # contrastive
                embeddings = model.get_embedding(images, actions_text)
                # Convert text -> label indices if needed
                # ...
                loss_contra = contrastive_loss(embeddings, action_labels)

                loss = loss_class + loss_contra

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        print(f"[Epoch {epoch+1}/{args.epochs}] Loss: {total_loss:.4f}")
    print('training complete')
    save_path = "embedding_model.pth"  # Current directory
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': args.epochs,
        'loss': total_loss,
    }, save_path)
    print(f"Model saved to {save_path}")


    
    # Extract embeddings for all demos in the folder
    embeddings_list = []
    values_list = []

    folder_path = args.path + "demo"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for folder_name in os.listdir(folder_path)[:15]:
        print(folder_name)
        sub_folder = os.path.join(folder_path, folder_name)
        
        if not os.path.isdir(sub_folder):
            continue
        
        action_name = folder_name.split("_")[-1]  # e.g., "3"
        
        action = action_description[int(action_name)]
    
        embedding = extract_embedding(model, sub_folder, action, device)
        embedding_array = embedding.numpy()
        embeddings_list.append(embedding_array)
        values_list.append(int(action_name))
    
    
    embeddings_array = np.array(embeddings_list)
    values_array = np.array(values_list)


    # Suppose embeddings_array.shape = (N, D) and values_array.shape = (N,)
    with h5py.File("known_embeddings.h5", "w") as f:
        f.create_dataset("embeddings", data=embeddings_array)
        f.create_dataset("values", data=values_array)


    print(f'saved embeddings')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train embeddings for a given robosuite-based task.")
    parser.add_argument("--task", type=str, required=True, help="Task name, e.g. 'can', 'square', etc.")
    parser.add_argument("--path", type=str, required=True, help="Path to dataset.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs.")
    parser.add_argument("--checkpoint_gradient", action="store_true", help="Enable gradient checkpointing.")
    args = parser.parse_args()

    train_embed(args)
