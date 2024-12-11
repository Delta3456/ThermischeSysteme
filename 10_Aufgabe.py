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
T_u = 20 + 273.15       # Umgebungstemperatur in K
eps = 0.5   # Absorptionsgrad
h = 5.0            # Konvektiver Wärmeübergang W/(m²·K)
sigma = 5.67e-8     # Stefan-Boltzmann-Konstante W/(m²·K^4)
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
dxy = 1e-4
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

def berechnung(P_l, Konvektion=False, endtime=60, dt=dt):
    # Zellenvariablen
    temp = fp.CellVariable(name="Temperature", mesh=mesh, value=T_u)
    quel_l = laser(P_l, x)
    las = fp.CellVariable(name="Laser", mesh=mesh, value=quel_l)

    eq = fp.TransientTerm(rho*cp) == fp.DiffusionTerm(lam) + las

