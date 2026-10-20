"""将完整声呐图与导航状态融合的 PPO CNN 特征提取器。"""
import torch
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class SonarCNNFeaturesExtractor(BaseFeaturesExtractor):
    """CNN 编码三路声呐，MLP 编码目标与速度，再拼接给 PPO actor/critic。"""

    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)
        sonar_shape = observation_space.spaces["sonar"].shape
        sonar_channels = sonar_shape[0]
        self.sonar_cnn = nn.Sequential(
            nn.Conv2d(sonar_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            sonar_features = self.sonar_cnn(
                torch.zeros(1, *sonar_shape)
            ).shape[1]

        state_dim = (observation_space.spaces["target"].shape[0]
                     + observation_space.spaces["velocity"].shape[0])
        self.state_mlp = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(sonar_features + 64, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations):
        # 声呐距离在环境里已归一化到 [0, 1]，状态向量保持物理数值。
        sonar_features = self.sonar_cnn(observations["sonar"])
        state = torch.cat((observations["target"], observations["velocity"]), dim=1)
        state_features = self.state_mlp(state)
        return self.fusion(torch.cat((sonar_features, state_features), dim=1))
