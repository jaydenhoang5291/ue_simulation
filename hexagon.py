import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon

# Tham số cơ bản
root3 = math.sqrt(3.0)
scale_factor = 3  # Yếu tố scale, có thể thay đổi (ví dụ: 1.0, 2.0, 0.5)

# Vector cơ bản sau khi scale
unitSlant = np.array([0.5, 0.5 * root3]) * scale_factor
unitHoriz = np.array([1.0, 0.0]) * scale_factor

# Tính bán kính hexagon
# Trong lưới hexagon, đường kính (2 * radius) phải bằng độ dài unitSlant
hex_radius = np.linalg.norm(unitSlant) / 2

# Tạo lưới hexagon
fig, ax = plt.subplots(1)
ax.set_aspect('equal')

# Phạm vi lưới (số hexagon theo m và n)
m_range = np.arange(-5, 5)
n_range = np.arange(-5, 5)

for m in m_range:
    for n in n_range:
        center = m * unitHoriz + n * unitSlant
        hex = RegularPolygon((center[0], center[1]), numVertices=6, radius=hex_radius*2/root3, 
                            orientation=0, edgecolor='black', facecolor='lightgray')
        ax.add_patch(hex)

# Đặt giới hạn trục chính xác
ax.set_xlim(min(m_range) - 20 * hex_radius, max(m_range) + 20 * hex_radius)           # ax.set_xlim(x_min, x_max)
ax.set_ylim(min(n_range) * 0.5 * root3 - 20 * hex_radius, max(n_range) * 0.5 * root3 + 20 * hex_radius)

# Hiển thị lưới
plt.grid(True)
plt.title(f'Hexagonal Grid (Scale Factor: {scale_factor})')
plt.show()