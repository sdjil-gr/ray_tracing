import numpy as np
import matplotlib.pyplot as plt


def cos_weighted_sample_hemisphere(n_samples, normal):
    points = []
    for _ in range(n_samples):
        u1 = np.random.rand()
        u2 = np.random.rand()

        phi = 2 * np.pi * u1
        theta = np.arccos(1.0 - 2.0 * u2)

        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)


        inn = (normal + np.array([0, 0, 1]))/np.linalg.norm(normal + np.array([0, 0, 1]))

        # 镜面对应样本到与法线垂直的方向
        x, y, z = 2.0 * np.dot(np.array([x, y, z]), inn) * inn - np.array([x, y, z]) 

        points.append((x, y, z))
    return np.array(points)


# 生成样本
n_samples = 5000
samples = cos_weighted_sample_hemisphere(n_samples, np.array([0, 0, 1]))

# 3D 可视化
fig = plt.figure(figsize=(12, 6))

# 绘制 3D 采样
ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(samples[:, 0], samples[:, 1], samples[:, 2], alpha=0.6, s=1)
ax1.set_xlabel('X axis')
ax1.set_ylabel('Y axis')
ax1.set_zlabel('Z axis')
ax1.set_title('Cos-Weighted Sampling on Hemisphere')

# 绘制 x-y 平面映射
ax2 = fig.add_subplot(122)
ax2.scatter(samples[:, 0], samples[:, 1], alpha=0.6, s=1)
ax2.set_xlabel('X axis')
ax2.set_ylabel('Y axis')
ax2.set_title('Projection onto XY Plane')
ax2.axis('equal')

plt.tight_layout()
plt.show()