"""
Dienstag, 8.10.24
@author Janik Focken
--------------------------------
Aufgabe 1 - Wärmeübertragung in einem Rohr
- Berechnung
- Ausgabe plotten

"""
import numpy as np
import matplotlib.pyplot as plt


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
    nu_a = np.where(
        (re >= 10000) & (pr >= 0.6) & (pr <= 160),  # Bedingung für gültige Reynolds-Zahl und Prandtl-Zahl
        0.023 * re ** (4 / 5) * pr ** n,  # Berechnung von Nusselt-Zahl nach a falls Bedingungen erfüllt
        np.nan  # NaN, falls Bedingungen nicht erfüllt
    )
    return nu_a


# Variablen 1a
re_array = np.linspace(10000, 2e5)  # Reynolds-Zahl Vektor
Prandtl_values = np.array([0.707, 5.83, 151])  # Prandtl-Zahl für Luft, Wasser, Ethylenglykol bei 300K
stoffe = ['Luft', 'Wasser', 'Ethylenglykol']  # Die dazugehörigen Stoffe

# Ausgabe plotten
plt.style.use('ggplot')
plt.figure()

# Berechnung der Nusselt-Zahl für jeden Stoff
for pr_value, stoff in zip(Prandtl_values, stoffe):
    Nu_a = nusselt_a(re_array, pr_value)
    plt.plot(re_array, Nu_a, label=f'{stoff} (Pr = {pr_value})')

plt.xlabel('Reynolds-Zahl')
plt.ylabel('Nusselt-Zahl')
plt.legend(title='Stoffe', loc='upper left')
plt.show()
