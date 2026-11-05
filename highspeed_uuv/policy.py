"""三路声呐 CNN 特征提取器与有界高斯 PPO 策略。"""
import torch
from torch import nn
from stable_baselines3.common.distributions import SquashedDiagGaussianDistribution
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.preprocessing import get_action_dim
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

        self.state_keys = tuple(
            key for key in ("target", "path", "velocity")
            if key in observation_space.spaces
        )
        state_dim = sum(observation_space.spaces[key].shape[0]
                        for key in self.state_keys)
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
        state = torch.cat(tuple(observations[key] for key in self.state_keys), dim=1)
        state_features = self.state_mlp(state)
        return self.fusion(torch.cat((sonar_features, state_features), dim=1))


class BoundedActorCriticPolicy(ActorCriticPolicy):
    """PPO policy with a tanh-squashed Gaussian over the normalized action box.

    Stable-Baselines3's default PPO Gaussian is unbounded and relies on the
    environment to clip actions. That makes stored log-probabilities disagree
    with the actions actually executed. This policy applies the tanh bijection
    inside the distribution and includes its log-probability correction.
    """

    def _build(self, lr_schedule):
        self.action_dist = SquashedDiagGaussianDistribution(
            get_action_dim(self.action_space)
        )
        super()._build(lr_schedule)
