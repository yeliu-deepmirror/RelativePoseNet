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
        average_size = 3
        self.features_process_l1 = nn.Sequential(
            nn.Conv2d(features_dim, 128, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
        )
        self.features_process_l1_post = nn.Sequential(
            nn.Conv2d(128, 32, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.AdaptiveAvgPool2d((average_size, average_size)),
            nn.Flatten(1, -1),
        )
        self.features_process_l2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.features_process_l2_post = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.AdaptiveAvgPool2d((average_size, average_size)),
            nn.Flatten(1, -1),
        )
        self.features_process_l3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 32, kernel_size=3, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.AdaptiveAvgPool2d((average_size, average_size)),
            nn.Flatten(1, -1),
        )
        self.pose_estimator = nn.Sequential(
            nn.Linear(32 * average_size * average_size * 3 + 8, 1024),
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
        features = self.features_process_l1(features)
        features_l1 = self.features_process_l1_post(features)

        features = self.features_process_l2(features)
        features_l2 = self.features_process_l2_post(features)

        features_l3 = self.features_process_l3(features)

        # add intrinsics and regress to pose
        features = torch.cat((features_l1, features_l2, features_l3, param1, param2), 1)
        pose = self.pose_estimator(features)

        return pose
