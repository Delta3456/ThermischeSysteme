"""
Dienstag, 05.11.24
@author Janik Focken
--------------------------------
Aufgabe 5 - Rohr + Verdichter
Teilaufgabe 1 - Diagramm Massen/Volumenstrom
Teilaufgabe 2 - benötigte Leistungen
Teilaufgabe 3 - Rohrdurchmesser
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import root

# Werte und Parameter
rho = 1.2041  # Luftdichte bei 20 °C in kg/m³
nu = 15.32e-6  # Kinematische Viskosität für Luft in m²/s
eta = 18.22e-6  # Dynamische Viskosität für Luft in kg/ms
L_rohr = np.linspace(1, 400, 100)  # Rohrlänge von 1 bis 400 Metern
diameters = [0.05, 0.1, 0.25, 0.5]  # Rohrdurchmesser in Metern


# Funktionen ---------------------------------------------------------------------------
# Funktion für die Druckdifferenz vom Verdichter (quadratischer Funktion, aus Aufgabe 04)
def pressure_difference(v_dot):
    return a * v_dot ** 2 + b * v_dot + c


# Funktion zur Berechnung des Darcy-Reibungsbeiwerts
def darcy_friction_factor(Re):
    if Re < 2300:
        return 16 / Re  # Laminar
    elif Re >= 2 * (10 ** 4):
        return 0.046 * Re ** -0.2
    else:
        return 0.079 * Re ** -0.25


# Funktion zur Berechnung des Volumenstroms
def berechne_volume_flow(D, L):
    A = np.pi * (D / 2) ** 2  # Querschnittsfläche des Rohrs
    Dh = (4 * A) / (2 * np.pi * (D / 2))  # Hydraulischer Durchmesser

    # Hilfsfunktion, im System gilt: delta_p_rohr + delta_p_verdichter = 0
    # (-dp_rohr/dx)=(f*2G^2)/(rho*Dh) und -dp_rohr = C*L_rohr
    # mit G=m_dot/A = rho*u
    def equation(v_dot):
        u = v_dot / A
        Re = Dh * u / nu
        f = darcy_friction_factor(Re)
        delta_p_rohr = L * (f * rho * u ** 2) / (2 * Dh)
        return pressure_difference(v_dot) - delta_p_rohr

    # Startwert für die Volumenstromsuche
    initial_guess = np.array([0.08])
    result = root(equation, initial_guess)

    # Überprüfen, ob die Lösung erfolgreich ist
    if result.success:
        return result.x[0]  # Gefundene Lösung für v_dot
    else:
        return np.nan  # Keine Lösung gefunden


# Berechnung und plotten ---------------------------------------------------------------------------
# Laden der Parameter aus der Excel-Datei
file_path = "04_parameter_speicherung.xlsx"
try:
    # Versuch, die Excel-Datei zu laden
    parameters_df = pd.read_excel(file_path)
    a, b, c = parameters_df["Druckdifferenz"]  # Parameter für Druckdifferenz aus der Excel-Datei extrahieren
    print("Excel-Datei wurde erfolgreich geladen.")
except FileNotFoundError:
    # Dieser Block wird ausgeführt, wenn die Datei nicht existiert
    print(f"Die Datei '{file_path}' existiert nicht.")
except Exception as e:
    # Allgemeiner Fehlerfall für andere Fehler
    print(f"Ein Fehler ist aufgetreten: {e}")

# Listen zur Speicherung der Ergebnisse
volume_flows_over_length = []
power_over_length = []

# Berechnungen für jeden Rohrdurchmesser
for D in diameters:
    volume_flow_for_length = []
    power_for_length = []

    # Berechnung für jede Rohrlänge
    for L in L_rohr:
        volume_flow = berechne_volume_flow(D, L)
        volume_flow_for_length.append(volume_flow)

        # Berechnung der Leistung, wenn ein Volumenstrom gefunden wurde
        if not np.isnan(volume_flow):
            delta_p = pressure_difference(volume_flow)
            power = delta_p * volume_flow
            power_for_length.append(power)
        else:
            power_for_length.append(np.nan)

    volume_flows_over_length.append(volume_flow_for_length)
    power_over_length.append(power_for_length)

# Plotten der Diagramme als Subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 6))

# Volumenstrom-Diagramm
for i, D in enumerate(diameters):
    ax1.plot(L_rohr, volume_flows_over_length[i], label=f'D = {D} m')
ax1.set_xlabel('Rohrlänge (m)')
ax1.set_ylabel('Erreichbarer Volumenstrom (m³/s)')
ax1.set_title('Erreichbarer Volumenstrom über Rohrlänge für verschiedene Rohrdurchmesser')
ax1.legend()
ax1.grid()

# Leistungs-Diagramm
for i, D in enumerate(diameters):
    ax2.plot(L_rohr, power_over_length[i], label=f'D = {D} m')
ax2.set_xlabel('Rohrlänge (m)')
ax2.set_ylabel('Benötigte Leistung (W)')
ax2.set_title('Benötigte Leistung über Rohrlänge für verschiedene Rohrdurchmesser')
ax2.legend()
ax2.grid()

plt.tight_layout()
plt.show()
