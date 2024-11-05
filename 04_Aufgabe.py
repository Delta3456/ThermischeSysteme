"""
Dienstag, 03.11.24
@author Janik Focken
--------------------------------
Aufgabe 4 - Datenanpassung
Teilaufgabe 1 - Funktion aus Daten
Teilaufgabe 2 - Parameter speichern
Teilaufgabe 3 - Berechnungen
Teilaufgabe 4 - maximal abgeführte Wärme
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Gegebene Daten (Abgelesen aus Datenblatt)
volume_flow_data = np.array([305, 250, 200, 150, 100, 90]) / 3600  # m³/h zu m³/s
pressure_diff_data = np.array([0, 75, 170, 250, 330, 340]) * 100  # mBar zu Pa
power_input_data = np.array([0.5, 1.2, 2.15, 2.85, 3.7, 3.75]) * 1000  # kW in W
temperature_increase_data = np.array([3, 8, 20, 42, 77, 83])  # °C

# Funktionen ---------------------------------------------------------------------------------
# Quadratische Funktion
def quadratic(x, a, b, c):
    return a * x ** 2 + b * x + c

def pressure_difference(v):
    return quadratic(v, *params_pressure)

def power_input(v):
    return quadratic(v, *params_power)

def temperature_increase(v):
    return quadratic(v, *params_temp)

# Aufgabe 1) Datenanpassung mit einer quadratischen Funktion
# Anpassung für Druckdifferenz, Leistungsaufnahme und Temperaturanstieg als Funktionen des Volumenstroms
# Kovarianzmatrix wird nicht genutzt, deswegen: _
params_pressure, _ = curve_fit(quadratic, volume_flow_data, pressure_diff_data)
params_power, _ = curve_fit(quadratic, volume_flow_data, power_input_data)
params_temp, _ = curve_fit(quadratic, volume_flow_data, temperature_increase_data)

# Aufgabe 2) Parameter speichern ----------------------------------------
parameter_data = {
    'Parameter': ['a', 'b', 'c'],
    'Druckdifferenz': params_pressure,
    'Leistungsaufnahme': params_power,
    'Temperaturanstieg': params_temp
}
parameter_df = pd.DataFrame(parameter_data)  # Erstellen eines DataFrames vom Dictionary
output_file = "parameter_speicherung.xlsx"
try:
    parameter_df.to_excel(output_file, index=False)  # DataFrame als Excel speichern und nur Daten speichern
    print(f"Parameter wurden in '{output_file}' gespeichert.")
except PermissionError:
    print(f"Fehler: Keine Berechtigung zum Speichern der Datei '{output_file}'")
except Exception as e:
    print(f"Fehler beim Speichern der Datei '{output_file}': {e}")

# Plotten der angepassten Funktionen (Aufgabe 1,2)) ---------------------------------------------------------
volume_flows = np.linspace(min(volume_flow_data), max(volume_flow_data), 100)
temperature_increases = [temperature_increase(v) for v in volume_flows]

plt.figure(figsize=(12, 12))

# Plot der Druckdifferenz
plt.subplot(2, 2, 1)
plt.plot(volume_flow_data, pressure_diff_data, 'o', label="Daten")
plt.plot(volume_flows, pressure_difference(volume_flows), '-', label="Anpassung")
plt.xlabel("Volumenstrom (m³/s)")
plt.ylabel("Druckdifferenz (Pa)")
plt.legend()
plt.grid()
plt.title("Druckdifferenz als Funktion des Volumenstroms")

# Plot der Leistungsaufnahme
plt.subplot(2, 2, 2)
plt.plot(volume_flow_data, power_input_data / 1000, 'o', label="Daten")
plt.plot(volume_flows, power_input(volume_flows) / 1000, '-', label="Anpassung")
plt.xlabel("Volumenstrom (m³/s)")
plt.ylabel("Leistungsaufnahme (kW)")
plt.legend()
plt.grid()
plt.title("Leistungsaufnahme als Funktion des Volumenstroms")

# Plot für Temperaturanstieg
plt.subplot(2, 1, 2)
plt.plot(volume_flow_data, temperature_increase_data, 'o', label="Daten")
plt.plot(volume_flows, temperature_increases, '-', label="Anpassung")
plt.xlabel("Volumenstrom (m³/s)")
plt.ylabel("Temperaturanstieg (°C)")
plt.legend()
plt.grid()
plt.title("Temperaturanstieg als Funktion des Volumenstroms")

plt.tight_layout()
plt.show()

# Aufgabe 3 und 4 ----------------------------------------------------------------------------------
k = 1.4  # Isentropenexponent für Luft

# Funktionen für die Aufgabe 3,4 -------------------------------------------
# Bei kleinen Druckänerungen kann diese Formel genutzt werden:
def isentropic_power(v):
    dP = pressure_difference(v)
    return v * dP * (k / (k - 1))

# Berechnung des Wirkungsgrads
def efficiency(v):
    P_isentropic = isentropic_power(v)
    P_actual = power_input(v)
    return P_isentropic / P_actual if P_actual > P_isentropic else 0

# Berechnung der Verluste
def losses(v):
    P_isentropic = isentropic_power(v)
    P_actual = power_input(v)
    return P_actual - P_isentropic if P_actual > P_isentropic else 0

volume_flows = np.linspace(min(volume_flow_data), max(volume_flow_data), 100)
isentropic_powers = [isentropic_power(v) for v in volume_flows]
efficiencies = [efficiency(v) for v in volume_flows]
losses_values = [losses(v) for v in volume_flows]

# Maximale Wärmeableitung als höchste Verlustleistung
max_heat_rejection = max(losses_values)

# Ausgabe der maximalen abgeführten Wärme
print(f"Maximal abzuführende Wärmeleistung (W): {max_heat_rejection}")

# Plot der Ergebnisse für Aufgabe 3 und 4
plt.figure(figsize=(12, 8))

# Plot der isentropen Leistung
plt.subplot(2, 2, 1)
plt.plot(volume_flows, np.array(isentropic_powers), label="Isentrope Leistung")
plt.xlabel("Volumenstrom (m³/s)")
plt.ylabel("Isentrope Leistung (W)")
plt.legend()
plt.title("Isentrope Leistung als Funktion der Druckdifferenz")

# Plot des isentropen Wirkungsgrades
plt.subplot(2, 2, 2)
plt.plot(volume_flows, efficiencies, label="Isentroper Wirkungsgrad")
plt.xlabel("Volumenstrom (m³/s)")
plt.ylabel("Isentroper Wirkungsgrad")
plt.legend()
plt.title("Isentroper Wirkungsgrad als Funktion der Druckdifferenz")

# Plot der Verluste
plt.subplot(2, 2, 3)
plt.plot(volume_flows, np.array(losses_values), label="Verluste")
plt.xlabel("Volumenstrom (m³/s)")
plt.ylabel("Verluste (W)")
plt.legend()
plt.title("Verluste als Funktion der Druckdifferenz")

# Darstellung der maximalen Wärmeableitung
plt.subplot(2, 2, 4)
plt.bar(["Maximale Wärmeableitung"], [max_heat_rejection], color="orange")
plt.ylabel("Maximale Wärmeableitung (W)")
plt.title("Maximal abgeführte Wärme")

plt.tight_layout()
plt.show()