import torch
import torch.nn as nn
from mamba_ssm import Mamba  # Assumed available
import vision_transformer as vits
from einops import rearrange

import torch
import torch.nn as nn
import torchvision.models as models

class SmallFlowEncoder(nn.Module):
    def __init__(self, output_dim=192):
        super().__init__()
        # Use pretrained or reduced ResNet (remove final FC)
        base_resnet = models.resnet18(pretrained=False)
        self.feature_extractor = nn.Sequential(*list(base_resnet.children())[:-1])  # [B, 512, 1, 1]
        self.fc = nn.Linear(512, output_dim)

    def forward(self, flow_seq):
        """
        Args:
            flow_seq: [T, 2, W, H] or [B, T, 2, W, H]
        Returns:
            embeddings: [T, output_dim] or [B, T, output_dim]
        """
        if flow_seq.dim() == 4:  # [T, 2, W, H]
            T = flow_seq.shape[0]
            x = self._prepare_input(flow_seq)  # [T, 3, W, H]
            features = self.feature_extractor(x).squeeze(-1).squeeze(-1)  # [T, 512]
            return self.fc(features)  # [T, output_dim]

        elif flow_seq.dim() == 5:  # [B, T, 2, W, H]
            B, T, _, W, H = flow_seq.shape
            flow_seq = flow_seq.reshape(B * T, 2, W, H)
            x = self._prepare_input(flow_seq)  # [B*T, 3, W, H]
            features = self.feature_extractor(x).squeeze(-1).squeeze(-1)  # [B*T, 512]
            features = self.fc(features).reshape(B, T, -1)  # [B, T, output_dim]
            return features

    def _prepare_input(self, flow):
        # Pad to 3 channels: [N, 2, W, H] → [N, 3, W, H]
        N, C, W, H = flow.shape
        if C == 2:
            pad = torch.zeros((N, 1, W, H), device=flow.device)
            flow = torch.cat([flow, pad], dim=1)
        return flow


class MaMoose(nn.Module):
    def __init__(self, embedding_dim=1024, hidden_dim=512, patch_hw=64, sequence_len=100):
        super(MaMoose, self).__init__()
        self.sequence_len = sequence_len
        self.patch_hw = patch_hw
        self.embedding_dim = embedding_dim

        # Flatten spatial patches and apply linear projection
        self.input_proj = nn.Linear(embedding_dim, hidden_dim)

        # Mamba operates on [B, L, D]
        self.mamba = Mamba(d_model=hidden_dim)

        # Output classifier: simple binary classification
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.motion_feature_extractor  = SmallFlowEncoder() #vits.__dict__['vit_tiny'](patch_size=2, num_classes=0, in_chans=2, img_size=(128, 128))
        
    def flow_encoder(self, optical_list):
        b, t = optical_list.shape[0], optical_list.shape[1]
        x = rearrange(optical_list, 'b t c w h -> (b t) c w h')
        of_embs = self.motion_feature_extractor(x)
        return of_embs

    def forward(self, optical_list, visual_list=None):
        """
        Args:
            optical_list: list of tensors [T, H, W, C], each with shape [100, 2, 224, 224]
        Returns:
            logits: [B, 1]
        """
        # B = len(optical_list)
        device = optical_list[0].device

        # Stack and sum inputs: shape [B, T, H, W, C]
        optical = optical_list  # [B, 100, 64, 64, 1024]
        # visual = torch.stack(visual_list)
        
        # b, t = optical.shape[0], optical.shape[1]
        # optical = rearrange(optical, 'b t c w h -> (b t) c w h')
        flow_embs = self.flow_encoder(optical)
        # flow_embs = rearrange(flow_embs, '(b t) e -> b t e', b=b, t=t)

        # Binary classification
        # logits = self.classifier(x)  # [B, 1]

        return flow_embs

model = MaMoose().cuda()
x = torch.randn(2, 20, 2, 128, 128).cuda()
y = model(x)
print(y.shape)