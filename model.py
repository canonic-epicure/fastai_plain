import numpy as np
import torch.nn as nn
import torch.optim
from transformers import AutoModelForImageClassification
from transformers.modeling_outputs import ImageClassifierOutput


#-----------------------------------------------------------------------------------------------------------------------
class AdaptiveConcatPool2d(nn.Module):
    "Layer that concats `AdaptiveAvgPool2d` and `AdaptiveMaxPool2d`"
    def __init__(self, size=1):
        super().__init__()

        self.size = size
        self.ap = nn.AdaptiveAvgPool2d(self.size)
        self.mp = nn.AdaptiveMaxPool2d(self.size)

    def forward(self, x):
        return torch.cat([self.mp(x), self.ap(x)], 1)


class BodyOfPretrained(nn.Module):
    def __init__(self, model_id):
        super().__init__()

        pretrained = AutoModelForImageClassification.from_pretrained(
            model_id,
            num_labels=0,
            ignore_mismatched_sizes=True
        )

        pretrained.timm_model.head.fc = nn.Identity()

        self.num_features = pretrained.config.num_features

        self.body = pretrained.timm_model

    def forward(self, x):
        return self.body.forward_features(x)


class MySimpleModel(nn.Module):
    def __init__(self, model_id, num_classes):
        super().__init__()

        self.num_classes = num_classes

        dict = self.dict = {}

        body = BodyOfPretrained(model_id)

        num_features = body.num_features

        dict['body'] = body

        dict['head_pool'] = AdaptiveConcatPool2d(size=1)
        dict['bnorm1'] = nn.BatchNorm1d(num_features * 2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        dict['lin1'] = nn.Linear(num_features * 2, 512, bias=False)
        dict['bnorm2'] = nn.BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        dict['lin2'] = nn.Linear(512, num_classes, bias=False)

        head = dict['head'] = nn.Sequential(
            dict['head_pool'],
            nn.Flatten(),
            dict['bnorm1'],
            nn.Dropout(p=0.25, inplace=False),
            dict['lin1'],
            nn.ReLU(inplace=True),
            dict['bnorm2'],
            nn.Dropout(p=0.5, inplace=False),
            dict['lin2'],
        )

        self.model = nn.Sequential(body, head)

    def forward(self, x):
        return ImageClassifierOutput(logits=self.model(x))
