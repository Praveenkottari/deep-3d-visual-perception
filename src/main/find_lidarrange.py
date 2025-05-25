import os
import numpy as np

lidar_dir = "/home/airl010/1_Thesis/visionNav/fusion/dataset/IISc_drive/velodyne_points/data"
all_bounds = []

for fname in os.listdir(lidar_dir):
    if fname.endswith(".bin"):
        pts = np.fromfile(os.path.join(lidar_dir, fname), dtype=np.float32).reshape(-1, 4)
        x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
        bounds = [x.min(), x.max(), y.min(), y.max(), z.min(), z.max()]
        all_bounds.append(bounds)

all_bounds = np.array(all_bounds)
global_min = all_bounds.min(axis=0)
global_max = all_bounds.max(axis=0)

print(f"Generalized boundary:")
print(f"minX: {global_min[0]:.2f}, maxX: {global_max[0]:.2f}")
print(f"minY: {global_min[2]:.2f}, maxY: {global_max[2]:.2f}")
print(f"minZ: {global_min[4]:.2f}, maxZ: {global_max[4]:.2f}")
