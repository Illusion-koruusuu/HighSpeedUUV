"""NED 坐标系下的刚体姿态积分，仅包含 HoloOcean Fossen 所需公式。"""
import numpy as np


def attitude_euler(eta, nu, dt):
    """用体坐标速度更新 NED 位置和欧拉角（z 轴正方向向下）。"""
    roll, pitch, yaw = eta[3:]
    sr, cr = np.sin(roll), np.cos(roll)
    sp, cp = np.sin(pitch), np.cos(pitch)
    sy, cy = np.sin(yaw), np.cos(yaw)

    rotation = np.array([
        [cp * cy, sr * sp * cy - cr * sy, cr * sp * cy + sr * sy],
        [cp * sy, sr * sp * sy + cr * cy, cr * sp * sy - sr * cy],
        [-sp, sr * cp, cr * cp],
    ])
    angle_rate = np.array([
        [1.0, sr * np.tan(pitch), cr * np.tan(pitch)],
        [0.0, cr, -sr],
        [0.0, sr / cp, cr / cp],
    ])
    eta[:3] += dt * rotation @ nu[:3]
    eta[3:] += dt * angle_rate @ nu[3:]
    return eta
