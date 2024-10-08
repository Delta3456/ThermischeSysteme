"""
Dienstag, 8.10.24
@author Janik Focken
--------------------------------
Aufgabe 1 - Wärmeübertragung in einem Rohr
- Berechnung
- Korrelation
- Print

"""
import numpy as np

# Stoffwerte bei 300K
Prandtl_values = {'Luft': 0.707, 'Wasser': 5.83, 'Ethylenglykol': 151}


# Funktion: Nusselt nach Aufgabe 1a
def nusselt_a(re, pr, n=1):
    """
    Funktion berechnet die Nusselt-Zahl nach Aufgabe 1a
    Bedingungen: turbulent, 0.6≤pr≤160, re≥10000
    -----
    :param re: Reynoldszahl
    :param pr: Prandtl-Zahl
    :param n: Exponent
    :return nu_a: Nusselt-Zahl
    """
    if re >= 10000:
        print("Die Reynoldszahl liegt außerhalb des gültigen Bereichs")
    elif pr <= 0.6 or pr >= 160:
        print("Die Prandtl-Zahl liegt außerhalb des gültigen Bereichs")
    else:
        nu_a = 0.023 * re ** (4 / 5) * pr ** n
    return nu_a

# Variablen
re_range = np.geomspace(10,2e5)

# Ploten der Funktionen
Nu1 = nusselt_a(re_range, Prandtl_values)