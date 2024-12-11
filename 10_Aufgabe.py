"""
Dienstag, 10.12.24
@author Janik Focken
--------------------------------
Aufgabe 10 - FiPy
"""
import matplotlib.pyplot as plt
import fipy as fp
import numpy as np

# Parameter
T_u = 20.0       # Umgebungstemperatur
eps = 0.5   # Absorptionsgrad
h = 5.0            # Konvektiver Wärmeübergang W/(m²·K)
sigma = 5.67e-8     # Stefan-Boltzmann-Konstante W/(m²·K^4)
endtime = 300.0
dt = 0.1 # Zeitschritt für die Brechnung

# Stoffdaten Stahl
rho = 8000.0       # Dichte Stahl kg/m^3
cp = 460.0         # Wärmekapazität J/(kg·K)
lam = 17.0         # Wärmeleitfähigkeit W/(m·K)

# Geometrie der Platte
L = 0.1            # Länge der Platte [m]
d = 0.005          # Dicke [m]
b = 0.02           # Breite [m]
laser_width = 0.005  # Breite der Laserzone [m]

# Gitter
dxy = 1e-3
nx = int(L / dxy) # Zellenanzahl in x-Richtung
mesh = fp.Grid1D(dx=d, nx=nx)
x = mesh.x.value


def laser(P_l, x):
    """
    Funktion zur Berechnung des volumetrischen Quellterms des Lasers
    """
    laser_center = 0.5 * L
    bereich = (x > laser_center - laser_width / 2) & (x < laser_center + laser_width / 2)
    quel_l = np.zeros(nx)
    quel_l[bereich] = (P_l * eps) / (d * b * laser_width)
    return quel_l

def berechnung(P_l, Konvektion=False, endtime=endtime, dt=dt):
    # Zellenvariablen
    temp = fp.CellVariable(name="Temperature", mesh=mesh, value=T_u)
    quel_l = laser(P_l, x)
    las = fp.CellVariable(name="Laser", mesh=mesh, value=quel_l)

    eq = fp.TransientTerm(rho*cp) == fp.DiffusionTerm(lam) + las

    # Hier noch Konvektion einfüfen
    # Lösen
    steps = int(endtime / dt)
    times = []
    max_temps = []
    T_1_grenz = None
    T_2_grenz = None
    for step in range(steps):
        current_time = (step + 1) * dt
        eq.solve(var=temp, dt=dt)

        T_max = temp.value.max()
        max_temps.append(T_max)
        times.append(current_time)

        # Erreichen bestimmter Temperaturen prüfen
        if (T_1_grenz is None) and (T_max >= 80.0):
            T_1_grenz = current_time
        if (T_2_grenz is None) and (T_max >= 180.0):
            T_2_grenz = current_time

    return np.array(times), np.array(max_temps), T_1_grenz, T_2_grenz, temp

# Verschiedene Szenarien simulieren
szenarien = [
    {"P": 1.0, "ko": False, "label": "1W ohne Konv."},
    #{"P": 1.0, "ko": True, "label": "1W mit Konv."},
    {"P": 50.0, "ko": False, "label": "50W ohne Konv."},
    #{"P": 50.0, "ko": True, "label": "50W mit Konv."},
]

plt.figure(figsize=(10, 6))
for sz in szenarien:
    times, max_temps, T_1_grenz, T_2_grenz, temp = berechnung(sz["P"], sz["ko"])
    plt.plot(times, max_temps, label=sz["label"])
    print(f"\nErgebnisse für {sz['label']}:")
    if T_1_grenz is not None:
        print(f"Zeit bis 80°C: {T_1_grenz:.2f} s")
    else:
        print(f"Zeit bis 80°C: Temperatur nicht erreicht, innerhalb von {endtime} s")
    if T_2_grenz is not None:
        print(f"Zeit bis 180°C: {T_2_grenz:.2f} s")
    else:
        print(f"Zeit bis 180°C: Temperatur nicht erreicht, innerhalb von {endtime} s")
    if 60 in times:
        idx = np.where(times == 60)[0][0]  # Ersten Treffer nehmen
        print(f"Max. Temperatur nach 60 s: {max_temps[idx]:.2f} °C")
    print("--------------------------------------------------------")

plt.xlabel("Zeit [s]")
plt.ylabel("Max. Temperatur [°C]")
plt.title("Maximale Temperaturentwicklung über die Zeit")
plt.legend()
plt.grid(True)
plt.show()