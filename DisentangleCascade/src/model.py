import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class HSICRegularizer(nn.Module):
    """Hilbert-Schmidt Independence Criterion with learnable bandwidth."""

    def __init__(self, sigma=5.0):
        super().__init__()
        self.log_sigma = nn.Parameter(torch.tensor(float(sigma)).log())

    @property
    def sigma(self):
        return self.log_sigma.exp()

    def _rbf_kernel(self, X, Y):
        dist = torch.cdist(X, Y, p=2) ** 2
        return torch.exp(-dist / (2 * self.sigma ** 2))

    def _center_kernel(self, K):
        n = K.size(0)
        H = torch.eye(n, device=K.device) - torch.ones(n, n, device=K.device) / n
        return H @ K @ H

    def forward(self, feature_list):
        total_hsic, n_pairs = 0.0, 0
        for i in range(len(feature_list)):
            for j in range(i + 1, len(feature_list)):
                X, Y = feature_list[i], feature_list[j]
                Kx = self._center_kernel(self._rbf_kernel(X, X))
                Ky = self._center_kernel(self._rbf_kernel(Y, Y))
                total_hsic += torch.trace(Kx @ Ky) / (X.size(0) ** 2)
                n_pairs += 1
        return total_hsic / n_pairs if n_pairs > 0 else torch.tensor(0.0, device=feature_list[0].device)


class UncertaintyEstimator(nn.Module):
    """MC Dropout-based epistemic uncertainty estimation."""

    def __init__(self, in_channels=256, dropout_p=0.3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 128, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.dropout = nn.Dropout2d(dropout_p)
        self.conv2 = nn.Conv2d(128, 1, 1)

    def forward(self, x, n_samples=5, compute_uncertainty=False):
        if not compute_uncertainty:
            h = F.relu(self.bn1(self.conv1(x)))
            return torch.sigmoid(self.conv2(h))

        preds = []
        for _ in range(n_samples):
            h = F.relu(self.bn1(self.conv1(x)))
            h = self.dropout(h)
            preds.append(torch.sigmoid(self.conv2(h)))
        return torch.var(torch.stack(preds, dim=0), dim=0)


class AttributeEncoder(nn.Module):
    """Architecturally-biased encoder for shape / color / texture."""

    def __init__(self, in_channels, out_channels, attr_type='shape'):
        super().__init__()
        if attr_type == 'shape':
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels), nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, 3, padding=1, groups=out_channels),
                nn.BatchNorm2d(out_channels), nn.ReLU(),
            )
        elif attr_type == 'color':
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels), nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels), nn.ReLU(),
            )
        else:  # texture
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels // 2, 3, padding=1, dilation=1),
                nn.BatchNorm2d(out_channels // 2), nn.ReLU(),
                nn.Conv2d(out_channels // 2, out_channels, 3, padding=2, dilation=2),
                nn.BatchNorm2d(out_channels), nn.ReLU(),
            )

    def forward(self, x):
        return self.conv(x)


class AdaptiveROIZoom(nn.Module):
    """Extract and re-encode high-uncertainty lesion regions."""

    def __init__(self, zoom_channels=512):
        super().__init__()
        self.zoom_encoder = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(256, zoom_channels, 3, padding=1), nn.BatchNorm2d(zoom_channels), nn.ReLU(),
            nn.AdaptiveAvgPool2d((7, 7)),
        )

    def forward(self, image, uncertainty_map, threshold=0.5):
        B, device = image.size(0), image.device
        zoomed = []
        for b in range(B):
            unc = uncertainty_map[b, 0]
            locs = (unc > threshold).nonzero(as_tuple=False)
            if len(locs) == 0:
                zoomed.append(torch.zeros(1, 512, 7, 7, device=device))
                continue
            y0, y1 = locs[:, 0].min().item(), locs[:, 0].max().item()
            x0, x1 = locs[:, 1].min().item(), locs[:, 1].max().item()
            margin = 0.2
            h, w = y1 - y0 + 1, x1 - x0 + 1
            y0 = max(0, int(y0 - h * margin)); y1 = min(6, int(y1 + h * margin))
            x0 = max(0, int(x0 - w * margin)); x1 = min(6, int(x1 + w * margin))
            s = 224 // 7
            roi = image[b:b+1, :, y0*s:(y1+1)*s, x0*s:(x1+1)*s]
            if roi.size(2) > 0 and roi.size(3) > 0:
                roi = F.interpolate(roi, size=(224, 224), mode='bilinear', align_corners=False)
                zoomed.append(self.zoom_encoder(roi))
            else:
                zoomed.append(torch.zeros(1, 512, 7, 7, device=device))
        return torch.cat(zoomed, dim=0)


class DisentangleCascade(nn.Module):
    """
    DisentangleCascade: Disentangled Representation Learning with
    Uncertainty-Driven ROI Zoom for Skin Lesion Classification.
    """

    def __init__(self, num_classes=8, backbone='resnet50', pretrained=True):
        super().__init__()
        resnet = models.resnet50(pretrained=pretrained)
        self.global_encoder = nn.Sequential(*list(resnet.children())[:-2])
        feat_dim = 2048

        self.global_proj = nn.Conv2d(feat_dim, 256, 1)
        self.uncertainty_head = UncertaintyEstimator(in_channels=256)
        self.roi_zoom = AdaptiveROIZoom(zoom_channels=512)

        combined = feat_dim + 512
        self.shape_encoder = AttributeEncoder(combined, 512, 'shape')
        self.color_encoder = AttributeEncoder(combined, 512, 'color')
        self.texture_encoder = AttributeEncoder(combined, 512, 'texture')

        self.hsic = HSICRegularizer(sigma=5.0)
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.fusion_controller = nn.Sequential(
            nn.Linear(512 * 3, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 3), nn.Softmax(dim=1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x, compute_uncertainty=False, return_features=False):
        local_feat = self.global_encoder(x)
        proj = self.global_proj(local_feat)
        unc_map = self.uncertainty_head(proj, compute_uncertainty=compute_uncertainty)
        zoom_feat = self.roi_zoom(x, unc_map)

        combined = torch.cat([local_feat, zoom_feat], dim=1)
        shape_v = self.gap(self.shape_encoder(combined)).flatten(1)
        color_v = self.gap(self.color_encoder(combined)).flatten(1)
        texture_v = self.gap(self.texture_encoder(combined)).flatten(1)

        hsic_loss = self.hsic([shape_v, color_v, texture_v])

        all_v = torch.cat([shape_v, color_v, texture_v], dim=1)
        w = self.fusion_controller(all_v)
        fused = w[:, 0:1] * shape_v + w[:, 1:2] * color_v + w[:, 2:3] * texture_v
        logits = self.classifier(fused)

        if return_features:
            return dict(logits=logits, hsic_loss=hsic_loss, uncertainty=unc_map,
                        shape=shape_v, color=color_v, texture=texture_v, importance=w)
        return logits, hsic_loss, unc_map
