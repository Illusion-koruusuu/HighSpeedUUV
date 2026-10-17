"""静态球形障碍物场景的三维 Dijkstra 代价场工具。

该文件只生成“当前位置到目标区域的自由空间代价”，用于强化学习奖励塑形，
不会直接输出需要执行的航迹或控制动作。
"""
from heapq import heappop, heappush

import numpy as np


def build_cost_field(goal, obstacles, start=(0.0, 0.0, 10.0), resolution=2.0,
                     margin=12.0, obstacle_margin=1.5, goal_radius=5.0,
                     planning_bounds=None, vertical_cost_scale=1.0):
    """构建从三维栅格到目标捕获区域的最小路径代价。

    代价场只用于奖励塑形。规划前会将障碍物半径额外膨胀，避免策略只优化
    “擦着障碍物表面通过”的危险路径。vertical_cost_scale 大于 1 时，
    垂向移动比等长的水平移动代价更高。
    """
    if resolution <= 0.0:
        raise ValueError("resolution must be positive")
    if vertical_cost_scale <= 0.0:
        raise ValueError("vertical_cost_scale must be positive")
    if planning_bounds is None:
        points = [np.asarray(start, dtype=float), np.asarray(goal, dtype=float)]
        for center, radius in obstacles:
            center = np.asarray(center, dtype=float)
            points.extend((center - radius, center + radius))
        bounds = np.asarray(points)
        lower = np.floor((bounds.min(axis=0) - margin) / resolution) * resolution
        upper = np.ceil((bounds.max(axis=0) + margin) / resolution) * resolution
    else:
        lower = np.asarray(planning_bounds[0], dtype=float)
        upper = np.asarray(planning_bounds[1], dtype=float)
        if lower.shape != (3,) or upper.shape != (3,) or np.any(upper <= lower):
            raise ValueError("planning_bounds must contain valid 3-D bounds")
        lower = np.floor(lower / resolution) * resolution
        upper = np.ceil(upper / resolution) * resolution
    shape = tuple(np.floor((upper - lower) / resolution).astype(int) + 1)

    # occupancy=True 表示该栅格不可通行。
    occupancy = np.zeros(shape, dtype=bool)
    # 生成 x、y、z 三个坐标轴及对应的三维栅格坐标。
    axes = [lower[i] + np.arange(shape[i]) * resolution for i in range(3)]
    grid_x, grid_y, grid_z = np.meshgrid(*axes, indexing="ij")
    for center, radius in obstacles:
        center = np.asarray(center, dtype=float)
        # 障碍物膨胀后再标记占用区域，给潜航器留出安全余量。
        inflated = float(radius) + obstacle_margin
        occupancy |= ((grid_x - center[0]) ** 2
                      + (grid_y - center[1]) ** 2
                      + (grid_z - center[2]) ** 2 <= inflated ** 2)

    def to_index(point):
        """将连续坐标映射到最近的栅格索引，并限制在边界内。"""
        index = np.rint((np.asarray(point, dtype=float) - lower) / resolution)
        return tuple(np.clip(index.astype(int), 0, np.asarray(shape) - 1))

    goal_index = to_index(goal)
    # 目标捕获球内的所有非障碍栅格都是终点，代价设为 0。
    goal_grid = ((grid_x - goal[0]) ** 2
                 + (grid_y - goal[1]) ** 2
                 + (grid_z - goal[2]) ** 2 <= goal_radius ** 2)
    goal_grid &= ~occupancy
    # 若目标球完全落在障碍物内，至少保留目标所在栅格作为可达终点。
    if not goal_grid.any():
        goal_grid[goal_index] = True
        occupancy[goal_index] = False

    # 从目标向外反向搜索：最终 costs[index] 是该栅格到目标的最小路径代价。
    costs = np.full(shape, np.inf, dtype=np.float32)
    queue = []
    for index in zip(*np.nonzero(goal_grid)):
        costs[index] = 0.0
        heappush(queue, (0.0, index))

    # 允许 26 邻域移动：直移、面对角线和体对角线的代价分别不同。
    neighbors = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if dx == dy == dz == 0:
                    continue
                offset = (dx, dy, dz)
                length = np.linalg.norm((dx, dy, vertical_cost_scale * dz))
                neighbors.append((offset, resolution * float(length)))

    # 多源 Dijkstra：目标捕获区域的所有栅格同时作为搜索起点。
    while queue:
        current_cost, current = heappop(queue)
        # 栅格代价用 float32 保存，而堆中的代价是 Python float；使用容差
        # 避免浮点舍入误差误删距离目标较远但仍有效的栅格。
        if current_cost > float(costs[current]) + 1e-5:
            continue
        for (dx, dy, dz), edge_cost in neighbors:
            neighbor = (current[0] + dx, current[1] + dy, current[2] + dz)
            if not (0 <= neighbor[0] < shape[0]
                    and 0 <= neighbor[1] < shape[1]
                    and 0 <= neighbor[2] < shape[2]):
                continue
            # 被障碍物占用的栅格不可通行，不能继续扩展。
            if occupancy[neighbor]:
                continue
            candidate = current_cost + edge_cost
            if candidate < costs[neighbor]:
                costs[neighbor] = candidate
                heappush(queue, (candidate, neighbor))

    return costs, lower, float(resolution), occupancy


def interpolate_cost(costs, origin, resolution, position):
    """在连续位置处对离散代价场做三线性插值。

    强化学习中的潜航器位置通常不在栅格中心，因此不能直接取最近栅格值。
    三线性插值可以让代价随位置连续变化，奖励也更平滑。
    """
    # 将世界坐标转换为以栅格为单位的连续坐标。
    coordinate = (np.asarray(position, dtype=float) - origin) / resolution
    # 离开规划区域时返回无穷大，由上层环境负责处理边界惩罚。
    if np.any(coordinate < 0.0) or np.any(coordinate > np.asarray(costs.shape) - 1):
        return float("inf")
    base = np.floor(coordinate).astype(int)
    fraction = coordinate - base
    value = 0.0
    weight_sum = 0.0
    # 遍历包围该位置的 8 个栅格顶点，按三维权重加权平均。
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                index = base + np.array([dx, dy, dz])
                if np.any(index >= np.asarray(costs.shape)):
                    continue
                weight = float(np.prod(np.where(
                    np.array([dx, dy, dz], dtype=bool), fraction, 1.0 - fraction
                )))
                cost = float(costs[tuple(index)])
                if np.isfinite(cost):
                    value += weight * cost
                    weight_sum += weight
    return value / weight_sum if weight_sum > 0.0 else float("inf")
