import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon

# Giả lập class để giữ cấu trúc code gốc
class CellularNetworkReceivedPower:
    def __init__(self):
        self.total_size = 10000
        self.grid_size = 50
        self.radius = 250  # Bán kính mỗi ô lục giác
        self.center = self.total_size / 2
        self.bs_colors = ['red', 'green', 'blue', 'cyan', 'magenta', 'yellow', 'black',
                         'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'lime',
                         'teal', 'navy', 'coral', 'gold', 'orchid']  # 19 màu cho 19 cụm
        self.fig = plt.figure(figsize=(10, 10))
        self.ax = self.fig.add_axes([0, 0, 1, 1])

    def setup_plot(self):
        # Vẽ lưới
        for i in range(self.grid_size + 1):
            x = i * (self.total_size / self.grid_size)
            self.ax.plot([x, x], [0, self.total_size], 'gray', linestyle=':', alpha=0.5)
            self.ax.plot([0, self.total_size], [x, x], 'gray', linestyle=':', alpha=0.5)

        # Tâm cụm trung tâm
        self.bs_positions = []
        center_positions = [(self.center, self.center)]  # BS0

        # Tính toán 6 tâm cụm xung quanh với khoảng cách đúng
        distance1 = np.sqrt(21) * self.radius  # Khoảng cách ~1145.75m
        for i in range(6):
            angle = 0.7721 * np.pi + i * 60 * np.pi / 180 
            cx = self.center + distance1 * np.cos(angle)
            cy = self.center + distance1 * np.sin(angle)
            center_positions.append((cx, cy))

        # Thêm 12 tâm cụm vòng 2
        # 6 tâm ở khoảng cách gấp đôi (~2291.5m)
        distance2 = 2 * np.sqrt(21) * self.radius  # Khoảng cách ~2291.5m
        for i in range(6):
            angle = 0.7721 * np.pi + i * 60 * np.pi / 180  # Góc: 0°, 30°, 60°, 90°, 120°, 150°
            cx = self.center + distance2 * np.cos(angle)
            cy = self.center + distance2 * np.sin(angle)
            center_positions.append((cx, cy))

        # 6 tâm còn lại ở khoảng cách tiếp xúc (~1579.25m)
        distance3 = 3*np.sqrt(7) * self.radius  # Khoảng cách ~1579.25m
        for i in range(6):
            angle = 0.7721 * np.pi + (i + 0.5) * 60 * np.pi / 180  # Góc: 15°, 45°, 75°, 105°, 135°, 165°
            cx = self.center + distance3 * np.cos(angle)
            cy = self.center + distance3 * np.sin(angle)
            center_positions.append((cx, cy))

        # Tạo các cụm lục giác
        bs_index = 0
        for cluster_idx, (cx, cy) in enumerate(center_positions):
            # Trạm gốc trung tâm của cụm với màu
            hex_center = RegularPolygon((cx, cy), numVertices=6,
                                      radius=self.radius, orientation=0,
                                      facecolor='none', edgecolor='blue')
            self.ax.add_patch(hex_center)
            self.bs_positions.append((cx, cy))
            self.ax.plot(cx, cy, marker='^', color=self.bs_colors[cluster_idx % len(self.bs_colors)], markersize=8)  # Màu cho tâm cụm
            bs_index += 1
            # 6 trạm gốc xung quanh với dấu chấm
            for i in range(6):
                angle = i * 60 * np.pi / 180
                x = cx + self.radius * np.sqrt(3) * np.cos(angle)
                y = cy + self.radius * np.sqrt(3) * np.sin(angle)
                hex_i = RegularPolygon((x, y), numVertices=6, radius=self.radius,
                                      orientation=0, facecolor='none', edgecolor='blue')
                self.ax.add_patch(hex_i)
                self.bs_positions.append((x, y))
                self.ax.plot(x, y, marker='.', color='black', markersize=5)  # Dấu chấm đen

    # Hàm tính tổng số ô lục giác
    def count_total_cells(self, bs_positions):
        return len(bs_positions)

# Tạo và chạy
network = CellularNetworkReceivedPower()
network.setup_plot()

# Tính và in tổng số ô
total_cells = network.count_total_cells(network.bs_positions)
print(f"Tổng số ô lục giác: {total_cells}")

# Thiết lập giới hạn trục
network.ax.set_xlim(0, network.total_size)
network.ax.set_ylim(0, network.total_size)
network.ax.set_xlabel('X (m)')
network.ax.set_ylabel('Y (m)')
network.ax.set_title('Hexagonal Grid with 19 Clusters')
network.ax.grid(True)

# Hiển thị
plt.show()