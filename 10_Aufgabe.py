"""
Dienstag, 10.12.24
@author Janik Focken
--------------------------------
Aufgabe 10 - FiPy
"""
import matplotlib.pyplot as plt
import fipy as fp
import numpy as np

# -----------------------------------------------------------
# Parameter
# -----------------------------------------------------------
T_u = 20.0 + 273.15 # Umgebungstemperatur
eps = 0.5 # Absorptionsgrad
alpha = 5.0 # Konvektiver Wärmeübergang
sigma = 5.67e-8 # Stefan-Boltzmann-Konstante
endtime = 120 # Gesamtberechnungszeit
dt = 0.5 # Zeitschritt für die Berechnung

# Stoffdaten Stahl
rho = 8000.0 # Dichte Stahl kg/m^3
cp = 460.0 # Wärmekapazität J/(kg·K)
lam = 17.0 # Wärmeleitfähigkeit W/(m·K)

# Geometrie der Platte
L = 0.1 # Länge der Platte m
d = 0.005 # Dicke m
b = 0.02 # Breite m
laser_width = 0.005  # Breite des Lasers m

# Gitter
dxy = 1e-3
nx = int(L / dxy) # Zellenanzahl in x-Richtung
mesh = fp.Grid1D(dx=dxy, nx=nx)
x = mesh.x.value

# -----------------------------------------------------------
# Funktionen
# -----------------------------------------------------------
def laser(P_l, x):
    """
    Berechnet den Volumen-Quellterm durch den Laser.
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

    if Konvektion:
        # Aus AB9_Laserheizen
        # Faktor von 2/d, damit Wärmestrom pro Volumen in der Berechnung steht
        q_verlust = - (2.0/ d) * (alpha * (temp - T_u) + sigma * eps * (temp ** 4 - T_u ** 4))
        eq = fp.TransientTerm(rho * cp) == fp.DiffusionTerm(lam) + las + q_verlust
    else:
        eq = fp.TransientTerm(rho*cp) == fp.DiffusionTerm(lam) + las

    steps = int(endtime / dt)
    times = []
    max_temps = []
    T_1_grenz = None
    T_2_grenz = None
    for step in range(steps):
        current_time = step * dt + dt
        eq.solve(var=temp, dt=dt)

        T_max = temp.value.max()
        max_temps.append(T_max)
        times.append(current_time)

        # Erreichen bestimmter Temperaturen prüfen
        if (T_1_grenz is None) and (T_max >= (80.0 + 273.15)):
            T_1_grenz = current_time
        if (T_2_grenz is None) and (T_max >= (180.0 + 273.15)):
            T_2_grenz = current_time

    return np.array(times), np.array(max_temps), T_1_grenz, T_2_grenz, temp

# -----------------------------------------------------------
# Berechnung und Plot
# -----------------------------------------------------------
# Verschiedene Szenarien simulieren
szenarien = [
    {"P": 1.0, "ko": False, "label": "1W ohne Konv./Stra."},
    {"P": 1.0, "ko": True, "label": "1W mit Konv./Stra."},
    {"P": 50.0, "ko": False, "label": "50W ohne Konv./Stra."},
    {"P": 50.0, "ko": True, "label": "50W mit Konv./Stra."},
]

# Ergebnisse speichern
end_temps = {}  # speichert Temperaturfelder am Ende

plt.figure(figsize=(10, 6))
for sz in szenarien:
    times, max_temps, T_1_grenz, T_2_grenz, temp = berechnung(sz["P"], sz["ko"])
    plt.plot(times, max_temps, label=sz["label"])
    end_temps[sz["label"]] = temp.value.copy()  # End-Temperaturfeld speichern
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
        print(f"Max. Temperatur nach 60 s: {max_temps[idx]:.2f} K")
    print("--------------------------------------------------------")

plt.xlabel("Zeit [s]")
plt.ylabel("Max. Temperatur [K]")
plt.title("Maximale Temperaturentwicklung über die Zeit")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10,6))
for label, T_feld in end_temps.items():
    plt.plot(x, T_feld, label=f"{label} nach {endtime}s")
plt.xlabel("Länge [m]")
plt.ylabel("Temperatur [K]")
plt.title("Temperaturverteilung über die Länge")
plt.grid(True)
plt.legend()
plt.show()