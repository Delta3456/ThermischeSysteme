"""
Dienstag, 22.10.24
@author Janik Focken
--------------------------------
Aufgabe 3 - Gleichungen lösen
Teilaufgabe 1 - Thermoelement
Teilaufgabe 2 - Verdampfer
"""
import numpy as np
from scipy.optimize import root
import matplotlib.pyplot as plt


# Funktionen --------------------------------------------------------------------
def thermoelement(t_te, epsilon):
    """
    Die Energiebilanz lautet:
    alpha * (T_f - t_e) = epsilon * sigma * (t_e^4 - T_u^4)
    ---
    :param t_te: Temperatur des Thermoelements (K)
    :param epsilon: Emissionsgrad (-)
    """
    thermoelement_wert = (alpha * (T_f - t_te)) - (epsilon * sigma * (t_te ** 4 - T_u ** 4))
    return thermoelement_wert


def berechne_t_te(epsilon):
    # Funktion zur Berechnung der Thermoelement-Temperatur für einen gegebenen Emissionsgrad
    loesung = root(thermoelement, t_te_schaetz, args=(epsilon,))
    if loesung.success:
        return loesung.x[0]
    else:
        print(f"Fehler!")
        return None

def w
# Teilaufgabe 1 --------------------------------------------------------------------
# Parameter
T_f = 1300  # Flammentemperatur in K
T_u = 300  # Umgebungstemperatur in K
alpha = 100  # Wärmeübergangskoeffizient W/(m^2*K)
sigma = 5.67e-8  # Stefan-Boltzmann-Konstante in W/(m^2*K^4)
t_te_schaetz = np.array([1000])  # Anfangsschätzwert für die Thermoelementtemperatur
epsilon_werte = np.linspace(0, 1, 21)  # Emissionsgrade von 0 bis 1 in Schritten von 0.05

# Berechnen von Temperaturwerte für verschiedene Emissionsgrade
t_te_werte = [berechne_t_te(epsilon) for epsilon in epsilon_werte]

# Plot erstellen
plt.figure(figsize=(8, 6))
plt.plot(epsilon_werte, t_te_werte)
plt.title('Temperatur des Thermoelements in Abhängigkeit vom Emissionsgrad')
plt.xlabel('Emissionsgrad (-)')
plt.ylabel('Temperatur des Thermoelements (K)')
plt.grid(True)
plt.show()
