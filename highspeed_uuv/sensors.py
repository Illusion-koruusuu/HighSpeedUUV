"""训练用的三路三维几何声呐。

这里不启动 Unreal，也不伪造 HoloOcean 的图像渲染；只使用与场景配置一致的
视场和量程，对当前训练中的球形障碍物做射线检测。三路输出统一栅格化为
角度-距离图；HoloOcean 原生侧扫声呐实际输出一维强度数组，部署时需转换适配。
"""
import numpy as np


class SonarArray:
    """前视 ImagingSonar 加左右 SidescanSonar 的三路视场。

    每路声呐输出两个通道：归一化最近距离和是否命中回波。三个视场沿通道
    维拼接为 ``(6, elevation_bins, azimuth_bins)``，交给 CNN 处理。
    """

    FRONT = "front"
    LEFT = "left"
    RIGHT = "right"

    def __init__(self, elevation_bins=24, azimuth_bins=64,
                 range_min=0.5, range_max=40.0):
        self.elevation_bins = int(elevation_bins)
        self.azimuth_bins = int(azimuth_bins)
        self.range_min = float(range_min)
        self.range_max = float(range_max)
        if self.elevation_bins <= 0 or self.azimuth_bins <= 0:
            raise ValueError("sonar bin counts must be positive")
        if not 0.0 <= self.range_min < self.range_max:
            raise ValueError("invalid sonar range")

        # 对应 HoloOcean 配置：前视 120° x 30°，左右侧视各 120° x 5°。
        self._views = {
            self.FRONT: (0.0, 120.0, 30.0),
            self.LEFT: (-90.0, 120.0, 5.0),
            self.RIGHT: (90.0, 120.0, 5.0),
        }
        self._angles = {
            name: (
                np.deg2rad(np.linspace(-elevation / 2.0, elevation / 2.0,
                                       self.elevation_bins)),
                np.deg2rad(np.linspace(yaw - azimuth / 2.0,
                                       yaw + azimuth / 2.0,
                                       self.azimuth_bins)),
            )
            for name, (yaw, azimuth, elevation) in self._views.items()
        }

    @property
    def observation_shape(self):
        return (6, self.elevation_bins, self.azimuth_bins)

    @staticmethod
    def _rotation_matrix(eta):
        """将艇体系方向转换到 NED 世界系；z 正方向向下。"""
        roll, pitch, yaw = eta[3:]
        sr, cr = np.sin(roll), np.cos(roll)
        sp, cp = np.sin(pitch), np.cos(pitch)
        sy, cy = np.sin(yaw), np.cos(yaw)
        return np.array([
            [cp * cy, sr * sp * cy - cr * sy, cr * sp * cy + sr * sy],
            [cp * sy, sr * sp * sy + cr * cy, cr * sp * sy - sr * cy],
            [-sp, sr * cp, cr * cp],
        ])

    def _view(self, eta, obstacles, view_name):
        elevation, azimuth = self._angles[view_name]
        origin = np.asarray(eta[:3], dtype=float)
        rotation = self._rotation_matrix(eta)
        # 一次构造整张角度网格，再批量求交，避免逐像素 Python 循环拖慢训练。
        elev_grid, azimuth_grid = np.meshgrid(elevation, azimuth, indexing="ij")
        cos_elevation = np.cos(elev_grid)
        local_directions = np.stack((
            cos_elevation * np.cos(azimuth_grid),
            cos_elevation * np.sin(azimuth_grid),
            np.sin(elev_grid),
        ), axis=-1)
        directions = (local_directions @ rotation.T).reshape(-1, 3)
        nearest = np.full(len(directions), self.range_max, dtype=np.float32)
        for center, radius in obstacles:
            offset = origin - np.asarray(center, dtype=float)
            b = directions @ offset
            c = float(np.dot(offset, offset) - radius * radius)
            discriminant = b * b - c
            valid = discriminant >= 0.0
            root = np.sqrt(np.maximum(discriminant, 0.0))
            near = -b - root
            far = -b + root
            distance = np.where(near >= 0.0, near, far)
            hits = valid & (distance >= self.range_min)
            nearest[hits] = np.minimum(nearest[hits], distance[hits])
        distances = nearest.reshape(self.elevation_bins, self.azimuth_bins)

        hit = (distances < self.range_max).astype(np.float32)
        normalized_distance = np.clip(
            (distances - self.range_min) / (self.range_max - self.range_min),
            0.0, 1.0,
        )
        return normalized_distance, hit

    def observe(self, eta, obstacles):
        """返回 CNN 使用的 six-channel sonar tensor。"""
        channels = []
        for view_name in (self.FRONT, self.LEFT, self.RIGHT):
            distance, hit = self._view(eta, obstacles, view_name)
            channels.extend((distance, hit))
        return np.asarray(channels, dtype=np.float32)

    def hit_count(self, sonar):
        """根据三个回波通道判断是否正在避障。"""
        return int(np.count_nonzero(sonar[[1, 3, 5]]))

    def near_obstacle(self, sonar, front_limit=18.0, side_limit=10.0):
        """只把近距离威胁标为避障，远处回波不应长期关闭巡航推进。

        前向只看中心 ±30°；侧向看左右声呐各自中心 ±35°。视场边缘、远距离
        障碍仍提供给 CNN，但不再被误判成当前必须减速/转向的威胁。
        """
        hit_channels = (1, 3, 5)
        distance_channels = (0, 2, 4)
        max_angle = 30.0
        angles = np.linspace(-60.0, 60.0, self.azimuth_bins)
        front_mask = np.abs(angles) <= max_angle
        side_mask = np.abs(angles) <= 35.0

        def close_return(view_index, angle_mask, limit):
            hit = sonar[hit_channels[view_index]][:, angle_mask] > 0.5
            normalized = sonar[distance_channels[view_index]][:, angle_mask]
            distance = self.range_min + normalized * (self.range_max - self.range_min)
            return bool(np.any(hit & (distance <= limit)))

        return (close_return(0, front_mask, front_limit)
                or close_return(1, side_mask, side_limit)
                or close_return(2, side_mask, side_limit))
