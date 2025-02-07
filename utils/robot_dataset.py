import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


class RobotDataset(Dataset):
    """
    Loads:
      - actions.txt (list of action indices)
      - success_rate.txt (list of success flags 0/1)
      - folder of demos (images) for each index: e.g. "demo1_action_0"
    Returns:
      Dictionary with { "action": <str>, "success": <int>, "images": <Tensor> }
    """
    def __init__(self, actions_file, success_file, image_folder, action_dict, transform=None):
        self.actions = np.loadtxt(actions_file, dtype=int)
        self.success = np.loadtxt(success_file).astype(int)
        self.image_folder = image_folder
        self.transform = transform
        self.action_dict = action_dict
        self.max_seq_len = 50

        if len(self.actions) != len(self.success):
            raise ValueError("Mismatch between number of actions and success labels.")

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, idx):
        action_idx = self.actions[idx]
        action_str = self.action_dict[action_idx]  # e.g. "Change the can color to red."
        success_flag = self.success[idx]
        
        # Example subfolder: "demo1_action_0"
        demo_folder = os.path.join(self.image_folder, f"demo{idx + 1}_action_{action_idx}")
        
        # Load all images from that folder
        image_filenames = os.listdir(demo_folder)
        images = []
        for img_name in image_filenames:
            img_path = os.path.join(demo_folder, img_name)
            with Image.open(img_path).convert("RGB") as img:
                if self.transform:
                    img = self.transform(img)
                images.append(img)

        # We store them in a fixed-size 3D tensor [seq_len, 3, H, W], truncated/padded
        if len(images) == 0:
            # Edge case: if no images exist (shouldn't happen ideally)
            # Return a tensor of zeros or handle gracefully
            padded_images = torch.zeros(self.max_seq_len, 3, 224, 224)
        else:
            # Create a zero-tensor
            padded_images = torch.zeros(self.max_seq_len, *images[0].shape)

            seq_len = min(len(images), self.max_seq_len)
            for i in range(seq_len):
                padded_images[i] = images[i]

        return {
            "action": action_str,
            "success": success_flag,
            "images": padded_images,
        }