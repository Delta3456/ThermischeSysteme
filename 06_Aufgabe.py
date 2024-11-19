"""
Dienstag, 12.11.24
@author Janik Focken
--------------------------------
Aufgabe 6 - Wasserkocher
Zwei Systeme definiert Heizelement und das Wasser.
"""
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Konstanten für System 1 (Heizelement)
alpha = 800  # Wärmeübergangskoeffizient (W/m²K)
A_h = np.pi * 0.009 * 0.6  # Oberfläche des Heizelements (Durchmesser 9 mm, Länge 60 cm)
c_v_h = 490  # Spezifische Wärmekapazität des Heizelements (J/kgK, Stahl)
dichte_stahl = 7850  # Dichte von Stahl in kg/m³
volumen_heizelement = np.pi * (0.009 / 2) ** 2 * 0.6  # Volumens des Heizelements
m_h = volumen_heizelement * dichte_stahl  # Masse Heizelement

# Konstanten für System 2 (Wasser)
volumen_w = 0.001  # Volumen in Kubikmetern (1 Liter)
durchmesser_w = 0.12  # Durchmesser in Metern (12 cm)
hoehe_w = volumen_w / (np.pi * (durchmesser_w / 2) ** 2)
A_aussen = np.pi * durchmesser_w * hoehe_w + 2 * (np.pi * (durchmesser_w / 2) ** 2)
alpha_aussen = 10  # Aussen anliegender Wärmeübergangskoeffizient (W/m²K)
epsilon = 0.9  # Emissionsgrad der Außenwand
sigma = 5.67e-8  # Stefan-Boltzmann-Konstante (W/m² K⁴)
T_u = 298  # Umgebungstemperatur (K)
m_w = 1  # Masse des Wassers (kg, 1 Liter)
c_v_w = 4200  # Spezifische Wärmekapazität von Wasser (J/kg K)


# Differentialgleichungen ---------------------------------------------------------------------
# DGL für System 1: Temperatur des Heizelements (T_h)
def dT_h_dt(t, T_h, T_w, P):
    Q_h = alpha * A_h * (T_w - T_h)
    return (P + Q_h) / (m_h * c_v_h)


# DGL für System 2: Temperatur des Wassers (T_w)
def dT_w_dt(t, T_w, T_h):
    Q_h = alpha * A_h * (T_w - T_h)
    Q_aussen = A_aussen * (T_u - T_w) * alpha_aussen + epsilon * sigma * (T_u ** 4 - T_w ** 4)
    return (Q_aussen - Q_h) / (m_w * c_v_w)


# DGL kombinieren
def system(t, y, P):
    T_h, T_w = y
    return [dT_h_dt(t, T_h, T_w, P), dT_w_dt(t, T_w, T_h)]


# Werte für die Berechnung
T_h0 = 298  # Anfangstemperatur des Heizelements
T_w0 = 298  # Anfangstemperatur des Wassers
y0 = [T_h0, T_w0]
t_span = (0, 500)  # Zeitspanne
t_eval = np.linspace(0, 500, 100)  # 100 Punkte Berechnen
powers = [1000, 2000]  # Leistungen für das Heizelement
solutions = {}

for P in powers:
    # Lösen der DGLs für jede Leistung
    # methode geändert, weil Runge-Kutta "wellen" verursacht hat
    solution = solve_ivp(system, t_span, y0, args=(P,), t_eval=t_eval, method='LSODA')
    solutions[P] = solution

# Plotten der Temperaturentwicklung für beide Leistungen
plt.figure(figsize=(12, 8))

for P, solution in solutions.items():
    zeit = solution.t
    T_h_loesung = solution.y[0]
    T_w_loesung = solution.y[1]
    plt.plot(zeit, T_h_loesung, label=f"Temperatur des Heizelements (T_h) bei {P/1000} kW", linestyle="-")
    plt.plot(zeit, T_w_loesung, label=f"Temperatur des Wassers (T_w) bei {P/1000} kW", linestyle="--")

# Siedetemperatur von Wasser hinzufügen
plt.axhline(373, color="red", linestyle=":", label="Siedetemperatur von Wasser (100°C)")

# Beschriftung
plt.xlabel("Zeit (Sekunden)")
plt.ylabel("Temperatur (K)")
plt.title("Temperaturentwicklung von Heizelement und Wasser über die Zeit bei verschiedenen Leistungen")
plt.legend()
plt.grid(True)
plt.show()
