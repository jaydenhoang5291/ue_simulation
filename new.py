import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon, Circle, Rectangle, Polygon
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
from matplotlib.widgets import Button
import pandas as pd

class CellularNetworkReceivedPower:
    def __init__(self):
        self.total_size = 10000
        self.grid_size = 50  # Số ô lưới chia khu vực
        self.center = self.total_size / 2
        self.ptx = 35  # Transmit Power in dBm
        self.gtx = 2  # Transmit Antenna Gain in dBi
        self.grx = 2  # Receive Antenna Gain in dBi
        self.sensitivity = -110  # Receiver sensitivity in dBm
        self.hom = 3  # Handover Margin in dB
        self.fc = 4  # Frequency in GHz
        self.current_serving_bs = None  # BS hiện tại phục vụ UE
        self.boundary = [
            (5000, 8000), (7590, 6490), (7596, 3499),
            (5000, 2000), (2400, 3500), (2400, 6500)
        ]
        self.velocity = 60
        self.velocity_ms = self.velocity * (1000 / 3600)  # km/h -> m/s
        self.steps = 60000  # Số bước mô phỏng
        self.time_per_step = 3
        self.distance_per_step = self.velocity_ms * self.time_per_step
        self.current_direction = 0  # 0: East, 1: North, 2: West, 3: South
        self.ue1_x = 5000
        self.ue1_y = 5000
        
        plt.ion()  # Bật chế độ tương tác của Matplotlib
        self.fig = plt.figure(figsize=(15, 12))
        self.ax = self.fig.add_axes([0.1, 0.1, 0.6, 0.8])
        self.table_ax = self.fig.add_axes([0.75, 0.4, 0.2, 0.3])
        self.table_ax.axis('off')
        self.toggle_ax = self.fig.add_axes([0.81, 0.02, 0.08, 0.04])
        self.restart_ax = self.fig.add_axes([0.91, 0.02, 0.08, 0.04])
        self.toggle_button = Button(self.toggle_ax, 'Stop', color='lightcoral')
        self.restart_button = Button(self.restart_ax, 'Restart', color='lightgreen')
        self.toggle_button.on_clicked(self.toggle_animation)
        self.restart_button.on_clicked(self.restart_animation)
        
        self.bs_positions = []  # Danh sách lưu vị trí BS
        # Tham số cho lưới lục giác
        self.root3 = math.sqrt(3.0)
        self.scale_factor = 500  # Yếu tố scale lưới lục giác
        self.unitSlant = np.array([0.5, 0.5 * self.root3]) * self.scale_factor
        self.unitHoriz = np.array([1.0, 0.0]) * self.scale_factor
        self.hex_radius = np.linalg.norm(self.unitSlant) / 2
        self.m_range = np.arange(-5, 6)  
        self.n_range = np.arange(-5, 6)  
        self.offset_x = self.center
        self.offset_y = self.center
        
        # Tạo màu ngẫu nhiên cho BS
        num_bs = len(self.m_range) * len(self.n_range)
        self.bs_colors = [tuple(np.random.rand(3)) for _ in range(num_bs)]
        
        self.animation_running = True
        self.current_frame = 0
        self.previous_serving_bs = None
        self.data_log = {
            'Step': [],
            'ue1_x': [],
            'ue1_y': [],
            'ue1_direction': [],
            'BS_connect': [],
            'Handover': [],
            'current_prx': []
        }
        
        self.setup_plot()  # Vẽ giao diện
        # Thêm cột prx vào data_log
        for i in range(len(self.bs_positions)):
            self.data_log[f'ue1_bs{i}_prx'] = []
    def calculate_distance(self, ue_x, ue_y, bs_x, bs_y):
        h_ut = 1.5
        d_2d = np.sqrt((ue_x - bs_x)**2 + (ue_y - bs_y)**2)
        d_3d = np.sqrt(d_2d**2 + h_ut**2)
        return d_3d

    def calculate_path_loss(self, d_3d):
        h_ut = 1.5
        return 13.54 + 39.08 * np.log10(d_3d) + 20 * np.log10(self.fc) - 0.6 * (h_ut - 1.5)

    def calculate_received_power(self, path_loss):
        return self.ptx + self.gtx + self.grx - path_loss

    def get_serving_bs(self, ue_x, ue_y):
        # Calculate distances to all BSs
        distances = [(i, self.calculate_distance(ue_x, ue_y, bs_x, bs_y)) for i, (bs_x, bs_y) in enumerate(self.bs_positions)]
        # Sort by distance and get the 6 nearest BSs
        distances.sort(key=lambda x: x[1])
        nearest_bs_indices = [d[0] for d in distances[:6]]
        
        received_powers = []
        for i in nearest_bs_indices:
            bs_x, bs_y = self.bs_positions[i]
            distance = self.calculate_distance(ue_x, ue_y, bs_x, bs_y)
            path_loss = self.calculate_path_loss(distance)
            received_power = self.calculate_received_power(path_loss)
            if received_power >= self.sensitivity:
                received_powers.append((i, received_power, distance, path_loss))
        
        if not received_powers:
            self.current_serving_bs = None
            return None, None, None
        received_powers.sort(key=lambda x: x[1], reverse=True)
        if self.current_serving_bs is None:
            best_bs, best_prx, best_distance, _ = received_powers[0]
            self.current_serving_bs = best_bs
            return best_bs, best_prx, best_distance
        current_bs_info = next((pl for pl in received_powers if pl[0] == self.current_serving_bs), None)
        if current_bs_info is None:
            best_bs, best_prx, best_distance, _ = received_powers[0]
            self.current_serving_bs = best_bs
            return best_bs, best_prx, best_distance
        current_bs_prx = current_bs_info[1]
        for bs_idx, bs_prx, bs_distance, _ in received_powers:
            if bs_prx > current_bs_prx + self.hom:
                self.current_serving_bs = bs_idx
                return bs_idx, bs_prx, bs_distance
        return self.current_serving_bs, current_bs_info[1], current_bs_info[2]

    def point_in_polygon(self, x, y, polygon):
        """Kiểm tra xem điểm (x, y) có nằm trong đa giác hay không bằng thuật toán ray-casting"""
        n = len(polygon)
        inside = False
        px, py = x, y
        for i in range(n):
            j = (i - 1) % n
            ax, ay = polygon[i]
            bx, by = polygon[j]
            if ((ay > py) != (by > py)) and (px < (bx - ax) * (py - ay) / (by - ay + 1e-10) + ax):
                inside = not inside
        return inside

    def line_segment_intersection(self, p1, p2, q1, q2):
        """Tính giao điểm của đoạn thẳng p1-p2 với q1-q2"""
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = q1
        x4, y4 = q2
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-10:  # Song song
            return None
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom
        if 0 <= t <= 1 and 0 <= u <= 1:
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)
            return (x, y)
        return None

    def find_intersection_and_edge(self, start, end):
        """Tìm giao điểm của đường UE di chuyển với biên lục giác và cạnh bị chạm"""
        min_t = float('inf')
        intersection = None
        edge_idx = -1
        for i in range(len(self.boundary)):
            j = (i + 1) % len(self.boundary)
            p = self.line_segment_intersection(start, end, self.boundary[i], self.boundary[j])
            if p:
                t = ((p[0] - start[0])**2 + (p[1] - start[1])**2) / ((end[0] - start[0])**2 + (end[1] - start[1])**2 + 1e-10)
                if 0 <= t < min_t:
                    min_t = t
                    intersection = p
                    edge_idx = i
        return intersection, edge_idx

    def setup_plot(self):
        for i in range(self.grid_size + 1):
            x = i * (self.total_size / self.grid_size)
            self.ax.plot([x, x], [0, self.total_size], 'gray', linestyle=':', alpha=0.5)
            self.ax.plot([0, self.total_size], [x, x], 'gray', linestyle=':', alpha=0.5)
        # Vẽ hình lục giác làm boundary
        hex_boundary = Polygon(self.boundary, fill=False, color='red', linestyle='--', linewidth=2)
        self.ax.add_patch(hex_boundary)
        
        # Tạo lưới hexagon sử dụng logic từ code đầu tiên
        self.bs_positions = []
        bs_index = 0
        for m in self.m_range:
            for n in self.n_range:
                center = m * self.unitHoriz + n * self.unitSlant + np.array([self.offset_x, self.offset_y])
                hex_patch = RegularPolygon((center[0], center[1]), numVertices=6, radius=self.hex_radius * 2 / self.root3, 
                                           orientation=0, edgecolor='blue', facecolor='none')
                self.ax.add_patch(hex_patch)
                # Đặt BS tại trung tâm mỗi hexagon
                self.bs_positions.append((center[0], center[1]))
                self.ax.plot(center[0], center[1], marker='^', color=self.bs_colors[bs_index], markersize=8, label=f'BS{bs_index}')
                bs_index += 1
        
        self.ax.set_xlim(0, self.total_size)
        self.ax.set_ylim(0, self.total_size)
        self.ax.text(self.total_size + self.total_size*0.005, 0, 'x',
                    ha='left', va='center', fontsize=12)
        self.ax.text(0, self.total_size + self.total_size*0.005, 'y',
                    ha='center', va='bottom', fontsize=12)
        self.ax.set_xlabel('Distance (m)')
        self.ax.set_ylabel('Distance (m)')
        self.ax.set_title('Cellular Network with Received Power Analysis')
        self.ue1_point, = self.ax.plot([], [], 'go', markersize=5, label='UE1')
        self.ue1_text = self.ax.text(0, 0, '', fontsize=8)
        self.ue1_line, = self.ax.plot([], [], '-', color='g', linewidth=2)
        self.time_text = self.ax.text(100, 2300, '', fontsize=10)
        self.ax.legend()
        plt.draw()

    def update_signal_table(self, ue1_x, ue1_y):
        self.table_ax.clear()
        self.table_ax.axis('off')
        # Calculate distances to all BSs
        distances = [(i, self.calculate_distance(ue1_x, ue1_y, bs_x, bs_y)) for i, (bs_x, bs_y) in enumerate(self.bs_positions)]
        distances.sort(key=lambda x: x[1])
        nearest_bs_indices = [d[0] for d in distances[:6]]
        
        ue1_received_powers = []
        for bs_idx in nearest_bs_indices:
            bs_x, bs_y = self.bs_positions[bs_idx]
            distance = self.calculate_distance(ue1_x, ue1_y, bs_x, bs_y)
            path_loss = self.calculate_path_loss(distance)
            received_power = self.calculate_received_power(path_loss)
            self.data_log[f'ue1_bs{bs_idx}_prx'].append(received_power)
            ue1_received_powers.append(f"{received_power:.1f}")
        
        table_data = [
            ['UE / BS', 'UE1'],
            *[[f'BS{i}', ue1_received_powers[j]] for j, i in enumerate(nearest_bs_indices)]
        ]
        table = self.table_ax.table(cellText=table_data, loc='upper left', cellLoc='center', colWidths=[0.15, 0.15])
        for cell in table._cells:
            cell_obj = table._cells[cell]
            cell_obj.set_edgecolor('black')
            if cell[0] == 0:
                cell_obj.set_facecolor('#E6E6E6')
                cell_obj.set_text_props(weight='bold')
        self.table_ax.set_title('Received Power (dBm) at 6 Nearest BS', pad=20, fontsize=9, fontweight='bold')
        self.table_ax.set_position([0.72, 0.2, 0.25, 0.7])

    def toggle_animation(self, event):
        self.animation_running = not self.animation_running
        if self.animation_running:
            self.toggle_button.label.set_text('Stop')
            self.toggle_button.color = 'lightcoral'
            self.run_animation()
        else:
            self.toggle_button.label.set_text('Continue')
            self.toggle_button.color = 'lightblue'
        self.toggle_button.ax.figure.canvas.draw()

    def restart_animation(self, event):
        self.animation_running = True
        self.current_frame = 0
        self.previous_serving_bs = None
        self.current_serving_bs = None
        self.data_log = {
            'Step': [],
            'ue1_x': [],
            'ue1_y': [],
            'ue1_direction': [],
            'BS_connect': [],
            'Handover': [],
            'current_prx': []
        }
        for i in range(len(self.bs_positions)):
            self.data_log[f'ue1_bs{i}_prx'] = []
        self.toggle_button.label.set_text('Stop')
        self.toggle_button.color = 'lightcoral'
        self.toggle_button.ax.figure.canvas.draw()
        self.run_animation()

    def save_data_to_csv(self):
        df = pd.DataFrame(self.data_log)
        for column in df.columns:
            if any(x in column for x in ['prx', '_x', '_y']):
                df[column] = df[column].round(1)
        bs_connect = df.pop('BS_connect')
        current_prx = df.pop('current_prx')
        handover = df.pop('Handover')
        df.insert(len(df.columns), 'BS_connect', bs_connect)
        df.insert(len(df.columns), 'current_prx', current_prx)
        df.insert(len(df.columns), 'Handover', handover)
        filename = f'received_power_data_{self.velocity}_timeperstep_{self.time_per_step}_ptx_{self.ptx}_stepsss_{self.steps}_hom_{self.hom}.csv'
        df.to_csv(filename, index=False)
        print(f"Data saved to {filename}")

    def update(self, frame):
        if not self.animation_running:
            return
        self.current_frame = frame
        old_direction = self.current_direction
        random_val = np.random.random()
        if random_val < 0.5:
            pass
        elif random_val < 0.75:
            self.current_direction = (self.current_direction + 1) % 4
        else:
            self.current_direction = (self.current_direction - 1 + 4) % 4
        if abs(self.current_direction - old_direction) == 2:
            self.current_direction = old_direction
        new_x = self.ue1_x
        new_y = self.ue1_y
        if self.current_direction == 0:  # Đông
            new_x = self.ue1_x + self.distance_per_step
        elif self.current_direction == 1:  # Bắc
            new_y = self.ue1_y + self.distance_per_step
        elif self.current_direction == 2:  # Tây
            new_x = self.ue1_x - self.distance_per_step
        else:  # Nam
            new_y = self.ue1_y - self.distance_per_step
        if self.point_in_polygon(new_x, new_y, self.boundary):
            self.ue1_x = new_x
            self.ue1_y = new_y
        else:
            start = (self.ue1_x, self.ue1_y)
            end = (new_x, new_y)
            intersection, edge_idx = self.find_intersection_and_edge(start, end)
            if intersection:
                self.ue1_x, self.ue1_y = intersection
                if edge_idx == 0 or edge_idx == 5:
                    self.current_direction = 3
                elif edge_idx == 2 or edge_idx == 3:
                    self.current_direction = 1
                elif edge_idx == 1:
                    self.current_direction = 2
                elif edge_idx == 4:
                    self.current_direction = 0
            else:
                self.current_direction = (self.current_direction + 1) % 4
        self.ue1_point.set_data([self.ue1_x], [self.ue1_y])
        bs1, Prx1, dist1 = self.get_serving_bs(self.ue1_x, self.ue1_y)
        handover = 0
        if self.previous_serving_bs is not None and bs1 != self.previous_serving_bs:
            handover = 1
        if bs1 is not None:
            bs_x, bs_y = self.bs_positions[bs1]
            self.ue1_text.set_position((self.ue1_x + 50, self.ue1_y + 50))
            self.ue1_text.set_text(f'UE1\nBS: {bs1}\nPrx: {Prx1:.1f} dBm\nDistance: {dist1:.1f}m')
            self.ue1_line.set_data([self.ue1_x, bs_x], [self.ue1_y, bs_y])
        else:
            self.ue1_text.set_text('')
            self.ue1_line.set_data([], [])
        self.update_signal_table(self.ue1_x, self.ue1_y)
        self.time_text.set_text(f'Step: {frame}')
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        self.data_log['Step'].append(frame)
        self.data_log['ue1_x'].append(self.ue1_x)
        self.data_log['ue1_y'].append(self.ue1_y)
        self.data_log['ue1_direction'].append(self.current_direction)
        self.data_log['BS_connect'].append(bs1)
        self.data_log['current_prx'].append(Prx1 if bs1 is not None else None)
        self.data_log['Handover'].append(handover)
        self.previous_serving_bs = bs1
        if frame == self.steps:
            self.save_data_to_csv()

    def run_animation(self):
        for frame in range(self.current_frame, self.steps + 1):
            if not self.animation_running:
                break
            self.update(frame)
            plt.pause(0.1)

if __name__ == "__main__":
    network = CellularNetworkReceivedPower()
    network.run_animation()
    plt.ioff()
    plt.show()