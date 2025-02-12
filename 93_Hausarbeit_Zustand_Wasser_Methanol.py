# Phasendiagramm

import numpy as np
import matplotlib.pyplot as plt
import CoolProp.CoolProp as CP
from CoolProp import AbstractState

# Erzeuge ein AbstractState-Objekt für das Wasser-Methanol-Gemisch (50:50 Molenbruch)
AS = AbstractState("HEOS", "Water&Methanol")
AS.set_mole_fractions([0.5, 0.5])  # 50:50 Molenbruch

# Kritische Eigenschaften in SI-Einheiten (Temperatur in Kelvin, Druck in Pascal)
T_crit_K = AS.T_critical()
p_crit_Pa = AS.p_critical()

# Umrechnung in Celsius und bar
T_crit_C = T_crit_K - 273.15       # von Kelvin in Celsius
p_crit_bar = p_crit_Pa / 1e5        # von Pascal in bar

print(f"Kritische Temperatur: {T_crit_C:.2f} °C")
print(f"Kritischer Druck: {p_crit_bar:.2f} bar")

# Definiere den Temperaturbereich (in °C) von 10°C bis 300°C
T_C = np.linspace(10, 300, 600)
P_bubble = np.empty_like(T_C)  # Array für Siedepunkt-Druckwerte (Q = 0)
P_dew = np.empty_like(T_C)     # Array für Taupunkt-Druckwerte (Q = 1)

# Schleife über den Temperaturbereich
for i, T in enumerate(T_C):
    T_K = T + 273.15  # Umrechnung von °C in Kelvin
    try:
        # Siedepunkt (erste Dampfblasenbildung in der Flüssigkeit): Q = 0
        AS.update(CP.QT_INPUTS, 0, T_K)
        P_bubble[i] = AS.p() / 1e5  # Druck in bar
    except Exception as e:
        P_bubble[i] = np.nan
        print(f"Fehler bei der Siedepunkt-Berechnung für T={T:.2f}°C: {e}")

    try:
        # Taupunkt (erste Tropfenbildung im Dampf): Q = 1
        AS.update(CP.QT_INPUTS, 1, T_K)
        P_dew[i] = AS.p() / 1e5  # Druck in bar
    except Exception as e:
        P_dew[i] = np.nan
        print(f"Fehler bei der Taupunkt-Berechnung für T={T:.2f}°C: {e}")

# Erstellen des Plots
plt.figure(figsize=(10, 6))
plt.plot(T_C, P_bubble, label="Siedelinie", color="blue", linewidth=2)
plt.plot(T_C, P_dew, label="Taulinie", color="red", linewidth=2)

# Kritischen Punkt als Marker einzeichnen
# plt.plot(T_crit_C, p_crit_bar, 'ko', label="Kritischer Punkt", markersize=10)

plt.xlabel("Temperatur (°C)", fontsize=12)
plt.ylabel("Druck (bar)", fontsize=12)
plt.title("Phasendiagramm eines Wasser-Methanol-Gemisches (50:50)", fontsize=14)
plt.xlim(10, 300)
plt.ylim(0.1, 20)  # Druckbereich ab 0,1 bar
plt.grid(True)
plt.legend(fontsize=12)
plt.show()