"""Cheap geometric range observations for fast training; not acoustic images.

HoloOcean supplies actual imaging and sidescan sonar during visual validation.
"""
import numpy as np


def _ray_sphere(origin, direction, center, radius):
    offset = origin - center
    b = np.dot(offset, direction)
    c = np.dot(offset, offset) - radius * radius
    discriminant = b * b - c
    if discriminant < 0:
        return np.inf
    root = np.sqrt(discriminant)
    near = -b - root
    far = -b + root
    return near if near >= 0 else far if far >= 0 else np.inf


class RangeSensors:
    """Front ±60° and side ±90° range fans in the horizontal plane."""
    def __init__(self, front_beams=61, side_beams=31, max_range=40.0):
        self.front_angles = np.deg2rad(np.linspace(-60, 60, front_beams))
        self.side_angles = np.deg2rad(np.r_[np.linspace(-120, -60, side_beams),
                                                np.linspace(60, 120, side_beams)])
        self.max_range = float(max_range)

    def _scan(self, eta, angles, obstacles):
        yaw = eta[5]
        origin = eta[:3]
        distances = np.full(len(angles), self.max_range, dtype=np.float32)
        for i, angle in enumerate(angles):
            direction = np.array([np.cos(yaw + angle), np.sin(yaw + angle), 0.0])
            for center, radius in obstacles:
                distance = _ray_sphere(origin, direction, np.asarray(center), radius)
                if distance < distances[i]:
                    distances[i] = distance
        return distances

    def observe(self, eta, obstacles):
        return {
            'front_range': self._scan(eta, self.front_angles, obstacles),
            'side_range': self._scan(eta, self.side_angles, obstacles),
        }
