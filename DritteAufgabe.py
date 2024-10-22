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


# Funktionen
def thermoelement(T, epsilon):
    """
    Funktion zur Energiebilanz des Thermoelements
    Energiebilanz: 0 = (A * alpha * (T_f - T)) - (A * epsilon * sigma * (T_u**4 - T_u**4))
    """
    thermoelement_wert = (alpha * (T_f - T)) - (epsilon * sigma * (T**4 - T_u**4))
    return thermoelement_wert

# Parameter
T_f = 1300  # Flammentemperatur in K
T_u = 300  # Umgebungstemperatur in K
alpha = 100  # Wärmeübergangskoeffizient W/(m^2*K)
sigma = 5.67e-8  # Stefan-Boltzmann-Konstante in W/(m^2*K^4)
epsilon = 0.5
# Teilaufgabe 1 ----------------------------------------------
solution = root(thermoelement, 600, args=(epsilon,))
print(solution)