import torch
import torch.nn as nn
import clip
from torch.nn.functional import normalize, relu
from torchvision.models import vit_b_16, ViT_B_16_Weights
from torch.utils.checkpoint import checkpoint
from torch.cuda.amp import autocast
import torch.nn.functional as F

class ViTCLIPModel(nn.Module):
    """
    Combines a ViT image encoder with a CLIP text encoder
    and outputs a binary classification (success vs fail)
    plus an embedding for contrastive learning.
    """
    def __init__(self, clip_model_name="ViT-B/32", latent_dim=512, checkpoint_gradient=False):
        super(ViTCLIPModel, self).__init__()

        # Vision Transformer for images
        self.image_encoder = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        # Replace final classifier layer with a trainable linear layer
        in_features = self.image_encoder.heads.head.in_features
        self.image_encoder.heads.head = nn.Linear(in_features, latent_dim)

        # CLIP model for text
        self.clip_model, _ = clip.load(clip_model_name)
        self.clip_model = self.clip_model.float()
        for param in self.clip_model.parameters():
            param.requires_grad = True  # Fine-tune text encoder?

        # Optionally enable gradient checkpointing
        self.checkpoint_gradient = checkpoint_gradient

        # Fusion and classification layers
        self.fc1 = nn.Linear(latent_dim * 2, 512)
        self.fc2 = nn.Linear(512, 1)

    def encode_images(self, image_sequence):
        """
        image_sequence: [batch_size, seq_len, 3, H, W]
        """
        batch_size, seq_len, c, h, w = image_sequence.size()
        # Flatten batch & seq together
        x = image_sequence.view(batch_size * seq_len, c, h, w)
        features = self.image_encoder(x)  # [batch_size*seq_len, latent_dim]
        # Reshape back and average over seq_len dimension
        features = features.view(batch_size, seq_len, -1).mean(dim=1)
        return features

    def forward(self, images, actions):
        """
        images:  [batch_size, seq_len, 3, H, W]
        actions: list of strings
        Returns:
          logit: shape [batch_size]
        """
        # -- encode images --
        if self.checkpoint_gradient:
            image_features = checkpoint(self.encode_images, images)
        else:
            image_features = self.encode_images(images)

        # -- encode text (CLIP) --
        text_tokens = clip.tokenize(actions).to(images.device)
        if self.checkpoint_gradient:
            text_features = checkpoint(self.clip_model.encode_text, text_tokens)
        else:
            text_features = self.clip_model.encode_text(text_tokens)

        # Combine them
        combined = torch.cat((image_features, text_features), dim=1)
        combined = F.normalize(combined, p=2, dim=1)

        x = torch.relu(self.fc1(combined))
        logit = self.fc2(x).squeeze(1)  # [batch_size]
        return logit

    def get_embedding(self, images, actions):
        """
        Return an embedding (post-fc1) used for contrastive learning.
        """
        with autocast(dtype=torch.float16):
            # Encode images
            image_features = self.encode_images(images)

            # Encode text
            text_tokens = clip.tokenize(actions).to(images.device)
            text_features = self.clip_model.encode_text(text_tokens)

            combined = torch.cat((image_features, text_features), dim=1)
            combined = F.normalize(combined, p=2, dim=1)
            final_embedding = torch.relu(self.fc1(combined))
        return final_embedding

