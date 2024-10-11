"""
Dienstag, 8.10.24
@author Janik Focken
--------------------------------
Aufgabe 1 - Wärmeübertragung in einem Rohr
"""
import numpy as np
import matplotlib.pyplot as plt


# Funktionen ------------------------------------------
def nusselt_a(re, pr, n=1):
    """
    Funktion berechnet die Nusselt-Zahl nach a
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


def nusselt_b(re, pr):
    """
    Funktion berechnet die Nusselt-Zahl nach b
    Bedingungen: turbulent, 0.5≤pr≤2000, 3000≤re≤5e6
    -----
    :param re: Reynoldszahl
    :param pr: Prandtl-Zahl
    :return nu_b: Nusselt-Zahl
    """
    f = (0.79 * np.log(re) - 1.64) ** (-2)

    nu_b = np.where(
        (re >= 3000) & (re <= 5e6) & (pr >= 0.5) & (pr <= 2000),  # Bedingung für gültige Reynolds-Zahl und Prandtl-Zahl
        # Berechnung von Nusselt-Zahl
        ((f / 8) * (re - 1000.) * pr) / (1 + 12.7 * (((f / 8) ** (1 / 2)) * (pr ** (2 / 3) - 1))),
        np.nan  # NaN, falls Bedingungen nicht erfüllt
    )
    return nu_b


def warmeuebergangskoef(nu, lam, d):
    """
    Berechnung des Wärmeübergangskoeffizienten
    Variablen:
    - nu: Nusselt-Zahl, [-]
    - d: Durchmesser [m]
    - lam: Wärmeleitfähigkeit [W/m K]
    - alpha: Wärmeübergangskoeffizient [W/m²K]
    Formel:
    a = (nu*lambda)/d
    """
    alpha = (nu * lam) / d
    return alpha


def re_rohr(u, d, v):
    """
    Berechnung der Reynolds-Zahl in Rohrströmung
    Variablen:
    - u: Geschwindigkeit [m/s]
    - d: Durchmesser [m]
    - v: Kinematische Viskosität [m²/s]
    - re: Reynolds-Zahl [-]
    Formel:
    - re = (u∗d)/v
    """
    re = (u * d) / v
    return re


# Skript Aufgabe 1a----------------------------------
# if __name__ == "__main__": # Alles was nicht in andere Module importiert werden soll hier hinter

# Variablen -------------------------------------
re_array = np.linspace(10000, 5e6)  # Reynolds-Zahl Vektor
Prandtl_values = np.array([0.707, 5.83, 151])  # Prandtl-Zahl für Luft, Wasser, Ethylenglykol bei 300K
stoffe = ['Luft', 'Wasser', 'Ethylenglykol']  # Die dazugehörigen Stoffe

# Aufgabe 1a plotten --------------------------------
plt.style.use('ggplot')
fig, axs = plt.subplots(1, 3)

# Berechnung der Nusselt-Zahl für jeden Stoff
for i, pr_value in enumerate(Prandtl_values):
    Nu_a_values = nusselt_a(re_array, pr_value)
    Nu_b_values = nusselt_b(re_array, pr_value)

    # Berechnung relativer Fehler
    relative_error = np.abs(Nu_b_values - Nu_a_values) / Nu_a_values
    max_error = np.nanmax(relative_error)  # nanmax ignoriert nan-werte
    # Ausgabe des maximalen Fehlers
    print(f"Prandtl-Zahl {pr_value}: Maximaler relativer Fehler = {max_error * 100:.2f}%")

    axs[i].plot(re_array, Nu_a_values, label=f'Nusselt a (Pr={pr_value})')
    axs[i].plot(re_array, Nu_b_values, label=f'Nusselt b (Pr={pr_value})')

    axs[i].set_xlabel('Reynolds-Zahl')
    axs[i].set_ylabel('Nusselt-Zahl')
    axs[i].set_xscale('log')
    axs[i].set_title(f'{stoffe[i]}')
    axs[i].legend(loc='best')

plt.tight_layout()
plt.show()

# Skript Aufgabe 1b----------------------------------
# Variablen
d = 0.03  # Durchmesser [m]
u = np.linspace(0.01, 100)  # Geschwindigkeit [m/s]
lam = np.array([26.3e-3, 613e-3, 252e-3])  # Wärmeleitfähigkeit [W/m K] (Luft, Wasser, Ethylenglykol)
v = np.array([15.89e-6, 8.5757e-7, 14.1e-6])  # Kin. Viskosität [m²/s] (Luft, Wasser, Ethylenglykol)