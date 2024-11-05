import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Gegebene Daten (Abgelesen aus Datenblatt)
volume_flow_data = np.array([305, 250, 200, 150, 100, 90]) / 3600  # m³/h zu m³/s
pressure_diff_data = np.array([0, 75, 170, 250, 330, 340]) * 100  # mBar zu Pa
power_input_data = np.array([0.5, 1.2, 2.15, 2.85, 3.7, 3.75]) * 1000  # kW in W
temperature_increase_data = np.array([3, 8, 20, 42, 77, 83])  # °C

# Quadratische Funktion zur Anpassung
def quadratic(x, a, b, c):
    return a * x ** 2 + b * x + c

# Anpassung der Funktionen für Druckdifferenz, Leistungsaufnahme und Temperaturanstieg in Abhängigkeit vom Volumenstrom
params_pressure, _ = curve_fit(quadratic, volume_flow_data, pressure_diff_data)
params_power, _ = curve_fit(quadratic, volume_flow_data, power_input_data)
params_temp, _ = curve_fit(quadratic, volume_flow_data, temperature_increase_data)

# Anpassungsfunktionen
def pressure_difference(v):
    return quadratic(v, *params_pressure)

def power_input(v):
    return quadratic(v, *params_power)

def temperature_increase(v):
    return quadratic(v, *params_temp)

# Adiabatenexponent und Umgebungstemperatur
k = 1.4  # für Luft
T_u = 293.15  # Umgebungstemperatur in Kelvin (20°C)

# Berechnung der isentropen Leistung in Abhängigkeit vom Volumenstrom
def isentropic_power(v):
    dP = pressure_difference(v)
    return v * dP * (k / (k - 1)) * T_u / T_u  # Einfache Darstellung, falls T_u konstant bleibt

# Berechnung des Wirkungsgrads
def efficiency(v):
    P_isentropic = isentropic_power(v)
    P_actual = power_input(v)
    return P_isentropic / P_actual if P_actual != 0 else 0

# Berechnung der Verluste
def losses(v):
    P_isentropic = isentropic_power(v)
    P_actual = power_input(v)
    return P_actual - P_isentropic if P_actual > P_isentropic else 0

# Berechnung für den Bereich des Volumenstroms
volume_flows = np.linspace(min(volume_flow_data), max(volume_flow_data), 100)
isentropic_powers = [isentropic_power(v) for v in volume_flows]
efficiencies = [efficiency(v) for v in volume_flows]
losses_values = [losses(v) for v in volume_flows]

# Maximale Wärmeableitung
max_heat_rejection = max(losses_values)

# Ausgabe der maximal abzuführenden Wärme
print(f"Maximal abzuführende Wärmeleistung (W): {max_heat_rejection}")

# Plot der Ergebnisse für Aufgabe 3
plt.figure(figsize=(12, 8))

# Plot der isentropen Leistung in Abhängigkeit vom Volumenstrom
plt.subplot(2, 2, 1)
plt.plot(volume_flows, isentropic_powers, label="Isentrope Leistung")
plt.xlabel("Volumenstrom (m³/s)")
plt.ylabel("Isentrope Leistung (W)")
plt.legend()
plt.title("Isentrope Leistung als Funktion des Volumenstroms")

# Plot des Wirkungsgrads in Abhängigkeit vom Volumenstrom
plt.subplot(2, 2, 2)
plt.plot(volume_flows, efficiencies, label="Isentroper Wirkungsgrad")
plt.xlabel("Volumenstrom (m³/s)")
plt.ylabel("Isentroper Wirkungsgrad")
plt.legend()
plt.title("Isentroper Wirkungsgrad als Funktion des Volumenstroms")

# Plot der Verluste in Abhängigkeit vom Volumenstrom
plt.subplot(2, 2, 3)
plt.plot(volume_flows, losses_values, label="Verluste")
plt.xlabel("Volumenstrom (m³/s)")
plt.ylabel("Verluste (W)")
plt.legend()
plt.title("Verluste als Funktion des Volumenstroms")

# Darstellung der maximalen Wärmeableitung
plt.subplot(2, 2, 4)
plt.bar(["Maximale Wärmeableitung"], [max_heat_rejection], color="orange")
plt.ylabel("Maximale Wärmeableitung (W)")
plt.title("Maximal abgeführte Wärme")

plt.tight_layout()
plt.show()