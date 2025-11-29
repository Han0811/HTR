import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align
from ultralytics import YOLO
from model import HTR_VT


class LineScore(nn.Module):
    def __init__(self, in_channels, hidden_dim=256):
        super(LineScore, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        x = self.pool(x).flatten(1)            # (N, C)
        x = torch.sigmoid(self.fc(x)).squeeze(-1)
        return x

class UnifiedHTR(nn.Module):
    def __init__(self,
                 yolo_weights="yolo11n.pt",
                 num_classes=96,
                 img_size=(32, 1024),
                 score_feat_channels=1):
        super().__init__()

        self.yolo = YOLO(yolo_weights)

        self.line_score = LineScore(
            in_channels=score_feat_channels,
            hidden_dim=256
        )

        self.htr = HTR_VT.create_model(
            nb_cls=num_classes,
            img_size=img_size
        )

        self.line_img_h, self.line_img_w = img_size

        self.score_conv = nn.Sequential(
            nn.Conv2d(1, score_feat_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )


    def crop_lines(self, image, boxes):
        if boxes is None or len(boxes) == 0:
            return None

        B, C, H, W = image.shape

        boxes_norm = boxes.clone().float()
        boxes_norm[:, [0, 2]] /= W
        boxes_norm[:, [1, 3]] /= H

        idxs = torch.zeros(len(boxes), dtype=torch.int64, device=image.device)

        crops = roi_align(
            image,
            [torch.cat([idxs.unsqueeze(1), boxes_norm], dim=1)],
            output_size=(self.line_img_h, self.line_img_w),
            aligned=True
        )
        return crops


    def forward(self, page_img, conf=0.35, iou=0.5):
        yolo_out = self.yolo(
            page_img,
            conf=conf,
            iou=iou,
            agnostic_nms=True,
            max_det=300,
            verbose=False
        )[0]

        if len(yolo_out.boxes) == 0:
            return None, None, None

        boxes = yolo_out.boxes.xyxy.to(page_img.device)  # (N, 4)
        crops = self.crop_lines(page_img, boxes)
        if crops is None:
            return boxes, None, None
        score_feat = torch.mean(crops, dim=1, keepdim=True)   # (N,1,H,W)
        score_feat = F.interpolate(score_feat, (32, 128))     # downsample
        score_feat = self.score_conv(score_feat)              # (N,Cs,32,128)

        score = self.line_score(score_feat)        # (N,)

        keep_idx = score > 0.5
        if keep_idx.sum() == 0:
            return boxes, score, None

        crops_kept = crops[keep_idx]               # (K,3,32,1024)

        logits = self.htr(crops_kept)              # (K, T, nb_cls)


        return boxes, score, logits
