"""
We have created a model with dropout layers in the classification and regression heads.
This is an extension of the SSD model from torchvision.

"""

import torch
from torch import nn
import torchvision
from torchvision.models.detection import _utils as det_utils
from torchvision.models.detection.ssd import (
    SSD, 
    DefaultBoxGenerator,
    SSDHead,
    _xavier_init,
    SSDClassificationHead,
    SSDRegressionHead
)
from typing import List, Dict


class SSDClassificationHeadWithDropout(SSDClassificationHead):
    def __init__(self, in_channels: List[int], num_anchors: List[int], num_classes: int, dropout_prob: float):
        super().__init__(in_channels, num_anchors, num_classes)
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, x):
        all_results = []

        for i, features in enumerate(x):
            results = self._get_result_from_module_list(features, i)
            results = self.dropout(results)  # Apply dropout
            N, _, H, W = results.shape
            results = results.view(N, -1, self.num_columns, H, W)
            results = results.permute(0, 3, 4, 1, 2)
            results = results.reshape(N, -1, self.num_columns)  # Size=(N, HWA, K)

            all_results.append(results)

        return torch.cat(all_results, dim=1)


class SSDRegressionHeadWithDropout(SSDRegressionHead):
    def __init__(self, in_channels: List[int], num_anchors: List[int], dropout_prob: float):
        super().__init__(in_channels, num_anchors)
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, x):
        all_results = []

        for i, features in enumerate(x):
            results = self._get_result_from_module_list(features, i)
            results = self.dropout(results)  # Apply dropout
            N, _, H, W = results.shape
            results = results.view(N, -1, self.num_columns, H, W)
            results = results.permute(0, 3, 4, 1, 2)
            results = results.reshape(N, -1, self.num_columns)  # Size=(N, HWA, K)

            all_results.append(results)

        return torch.cat(all_results, dim=1)


class SSDHeadWithDropout(SSDHead):
    def __init__(self, in_channels: List[int], num_anchors: List[int], num_classes: int, dropout_prob: float):
        super().__init__(in_channels, num_anchors, num_classes)
        self.classification_head = SSDClassificationHeadWithDropout(in_channels, num_anchors, num_classes, dropout_prob)
        self.regression_head = SSDRegressionHeadWithDropout(in_channels, num_anchors, dropout_prob)

    def forward(self, x):
        return {
            "bbox_regression": self.regression_head(x),
            "cls_logits": self.classification_head(x),
        }

def create_dropout_model(num_classes=91, size=300, nms=0.45, dropout_rate=0.1):
    model_backbone = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
    backbone = nn.Sequential(
        model_backbone.conv1, model_backbone.bn1, model_backbone.relu, model_backbone.maxpool,
        model_backbone.layer1, model_backbone.layer2, model_backbone.layer3, model_backbone.layer4
    )
    out_channels = [2048] * 6  # Example adjustment, ensure this matches your design
    anchor_generator = DefaultBoxGenerator([[2], [2, 3], [2, 3], [2, 3], [2], [2]])

    num_anchors = anchor_generator.num_anchors_per_location()
    head = SSDHeadWithDropout(out_channels, num_anchors, num_classes, dropout_rate)

    model = SSD(
        backbone=backbone,
        num_classes=num_classes,
        anchor_generator=anchor_generator,
        size=(size, size),
        head=head,
        nms_thresh=nms
    )
    return model


if __name__ == '__main__':
    model = create_dropout_model(2, 300)
    print(model)
    # Total parameters and trainable parameters.
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.")