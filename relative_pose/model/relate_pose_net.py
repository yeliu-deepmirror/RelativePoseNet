import torch
import torch.nn as nn
import torch.nn.functional as F
from .extractor import SmallEncoder


# reference : https://github.com/princeton-vl/RAFT/blob/master/core/raft.py
class RelativePoseNet(nn.Module):
    def __init__(self, dropout=0.0, feature_dim=128):
        super(RelativePoseNet, self).__init__()
        self.fnet = SmallEncoder(output_dim=feature_dim, norm_fn='instance', dropout=dropout)
        features_dim = feature_dim * 2
        self.features_process = nn.Sequential(
            nn.Conv2d(features_dim, 128, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 64, kernel_size=3, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 32, kernel_size=3, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.average_pool = nn.AdaptiveAvgPool2d((6, 6))
        self.pose_estimator = nn.Sequential(
            nn.Linear(32 * 6 * 6 + 8, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 7),
        )

    def forward(self, image1, param1, image2, param2):
        """ Estimate relative pose between pair of frames """
        image1 = 2 * (image1 / 255.0) - 1.0
        image2 = 2 * (image2 / 255.0) - 1.0

        image1 = image1.contiguous()
        image2 = image2.contiguous()

        # run the feature network
        fmap1, fmap2 = self.fnet([image1, image2])

        # concate the features
        features = torch.cat((fmap1, fmap2), 1)
        features = self.features_process(features)
        features = self.average_pool(features)
        features = torch.flatten(features, 1)

        # add intrinsics and regress to pose
        features = torch.cat((features, param1, param2), 1)
        pose = self.pose_estimator(features)

        return pose
