# Daten aus Grafik für Erdbodentemperatur
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline
from matplotlib.animation import FuncAnimation

# Originaldaten für Winter
winter_x = [
    10, 10.1967, 10.1941, 10.178, 10.1632, 10.1647, 10.1975, 10.2765, 10.4167,
    10.6224, 10.8488, 11.0362, 11.1344, 11.1183, 10.9669, 10.6617, 10.2104,
    9.6372, 8.9665, 8.2224, 7.4233, 6.5837, 5.7185, 4.8422, 3.9694, 3.1148
]
winter_y = [
    -20, -19.682, -18.6907, -17.6879, -16.6783, -15.6667, -14.6578, -13.6565,
    -12.6675, -11.6924, -10.719, -9.7304, -8.7196, -7.7052, -6.7105, -5.7581,
    -4.8667, -4.0522, -3.3304, -2.7154, -2.1988, -1.7576, -1.3686, -1.0086,
    -0.6544, -0.2827
]

# Originaldaten für Frühjahr
spring_x = [
    10, 10.2623, 10.1364, 10.0976, 10.1243, 10.1947, 10.287, 10.3811, 10.4575,
    10.4965, 10.4771, 10.3789, 10.1883, 9.8994, 9.5256, 9.0864, 8.5929, 8.0493,
    7.496, 7.0531, 6.8494, 7.0013, 7.5323, 8.306, 9.1688, 9.9672
]
spring_y = [
    -20, -19.7173, -18.7796, -17.8381, -16.8951, -15.9527, -15.0127, -14.0742,
    -13.1339, -12.1898, -11.2453, -10.306, -9.3835, -8.4911, -7.6347, -6.8171,
    -6.0377, -5.2921, -4.5525, -3.7384, -2.7985, -1.8821, -1.1356, -0.5731,
    -0.1945, 0
]

# Sommer und Herbst spiegeln
summer_x = [10 - (temp - 10) for temp in winter_x]
autumn_x = [10 - (temp - 10) for temp in spring_x]
summer_y = winter_y
autumn_y = spring_y

# Glätten der Daten
def smooth_data(x, y, num_points=100):
    spline = make_interp_spline(y, x, k=3)
    y_smooth = np.linspace(min(y), max(y), num_points)
    x_smooth = spline(y_smooth)
    return x_smooth, y_smooth

# Smoothing
winter_x_smooth, winter_y_smooth = smooth_data(winter_x, winter_y)
spring_x_smooth, spring_y_smooth = smooth_data(spring_x, spring_y)
summer_x_smooth, summer_y_smooth = smooth_data(summer_x, summer_y)
autumn_x_smooth, autumn_y_smooth = smooth_data(autumn_x, autumn_y)

# Interpolierte Übergänge
frames_per_transition = 20  # Reduziert für schnellere Übergänge

def interpolate_seasonal_data(x1, y1, x2, y2, steps):
    interpolated_x = np.zeros((steps, len(x1)))
    interpolated_y = np.zeros((steps, len(y1)))
    for i in range(steps):
        alpha = i / (steps - 1)
        interpolated_x[i] = (1 - alpha) * np.array(x1) + alpha * np.array(x2)
        interpolated_y[i] = (1 - alpha) * np.array(y1) + alpha * np.array(y2)
    return interpolated_x, interpolated_y

# Übergänge erstellen
winter_to_spring_x, winter_to_spring_y = interpolate_seasonal_data(winter_x_smooth, winter_y_smooth, spring_x_smooth, spring_y_smooth, frames_per_transition)
spring_to_summer_x, spring_to_summer_y = interpolate_seasonal_data(spring_x_smooth, spring_y_smooth, summer_x_smooth, summer_y_smooth, frames_per_transition)
summer_to_autumn_x, summer_to_autumn_y = interpolate_seasonal_data(summer_x_smooth, summer_y_smooth, autumn_x_smooth, autumn_y_smooth, frames_per_transition)
autumn_to_winter_x, autumn_to_winter_y = interpolate_seasonal_data(autumn_x_smooth, autumn_y_smooth, winter_x_smooth, winter_y_smooth, frames_per_transition)

# Frames zusammenfügen
all_x = np.concatenate([winter_to_spring_x, spring_to_summer_x, summer_to_autumn_x, autumn_to_winter_x])
all_y = np.concatenate([winter_to_spring_y, spring_to_summer_y, summer_to_autumn_y, autumn_to_winter_y])

# Animation erstellen
fig, ax = plt.subplots(figsize=(10, 8))
ax.set_xlim(0, 20)
ax.set_ylim(-20, 0)
ax.set_xlabel("Erdbodentemperatur (°C)")
ax.set_ylabel("Tiefe (m)")
ax.grid(True)
line, = ax.plot([], [], color='black')

def update(frame):
    x_data = all_x[frame]
    y_data = all_y[frame]
    line.set_data(x_data, y_data)
    return line,

# Animationseinstellungen: fps erhöht für schnellere Wiedergabe
num_frames = len(all_x)
ani = FuncAnimation(fig, update, frames=num_frames, blit=True)

# Speichern als GIF mit Pillow
output_path = "schnellere_erdbodentemperatur_animation.gif"
ani.save(output_path, writer='pillow', fps=40)  # fps auf 40 erhöht

plt.close(fig)

print("Animation gespeichert unter:", output_path)