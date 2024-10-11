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


def nusselt_b(re, pr, n=1):
    """
    Funktion berechnet die Nusselt-Zahl nach b
    Bedingungen: turbulent, 0.5≤pr≤2000, 3000≤re≤5e6
    -----
    :param re: Reynoldszahl
    :param pr: Prandtl-Zahl
    :param n: Exponent
    :return nu_a: Nusselt-Zahl
    """
    f=(0.79 * np.log(re) - 1.64)**(-2)

    nu_b = np.where(
        (re >= 3000) & (re <= 5e6) & (pr >= 0.5) & (pr <= 2000),  # Bedingung für gültige Reynolds-Zahl und Prandtl-Zahl
        ((f/8)*(re-1000.)*pr)/(1+12.7*(((f/8)**(1/2))*(pr**(2/3)-1))),  # Berechnung von Nusselt-Zahl nach b falls Bedingungen erfüllt
        np.nan  # NaN, falls Bedingungen nicht erfüllt
    )
    return nu_b


# Variablen 1a
re_array = np.linspace(3000, 5e6)  # Reynolds-Zahl Vektor
Prandtl_values = np.array([0.707, 5.83, 151])  # Prandtl-Zahl für Luft, Wasser, Ethylenglykol bei 300K
stoffe = ['Luft', 'Wasser', 'Ethylenglykol']  # Die dazugehörigen Stoffe

# Ausgabe plotten
plt.style.use('ggplot')
fig, axs = plt.subplots(1, 3)

# Berechnung der Nusselt-Zahl für jeden Stoff
for i, pr_value  in enumerate(Prandtl_values):
    Nu_a = nusselt_a(re_array, pr_value)
    Nu_b = nusselt_b(re_array, pr_value)

    axs[i].plot(re_array, Nu_a, label=f'Nusselt a (Pr={pr_value})')
    axs[i].plot(re_array, Nu_b, label=f'Nusselt b (Pr={pr_value})')

    axs[i].set_xlabel('Reynolds-Zahl')
    axs[i].set_ylabel('Nusselt-Zahl')
    axs[i].set_title(f'{stoffe[i]}')
    axs[i].legend(loc='upper left')

plt.subplots_adjust(wspace=1)
plt.show()
