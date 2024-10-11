# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 13:35:42 2021

@author: Christoph Horn
_______________________________________________________________________________
_______________________________________________________________________________

Skript: Aufgabe 1, Thermische Systeme



_______________________________________________________________________________
_______________________________________________________________________________
Beschreibung:


    - Wärmeübergang in Rohren
    - Vergleich von verschiedenen Korrelationen für die Nusselt-Zahl
    - Plots der Ergebnisse


_______________________________________________________________________________
_______________________________________________________________________________


"""

### Import der erforderlichen Module ###

###############################################################################

import numpy as np
import matplotlib.pyplot as plt


### Funktionen ###

###############################################################################


def reynolds_zahl(u: float, d: float, v: float):
    """
    Berechnung der Reynolds-Zahl


    Variablen:

    - u : Geschwindigkeit [m/s]
    - d : Durchmesser [m]
    - v : Kinematische Viskosität [m²/s]
    - re: Reynolds-Zahl [-]


    Formel:

        re = (u*d)/v


    """

    re = (u * d) / v

    return re


reynolds_zahl = np.vectorize(reynolds_zahl)


###############################################################################


def nusselt_zahl_1(re: float, pr: float):
    """
    Berechnung der Nusselt-Zahl nach Korrelation 1


    Variablen:

    - re : Reynolds-Zahl [-]
    - pr : Prandtl-Zahl [-]
    - nu : Nusselt-Zahl [-]
    - f  : Faktor [-]


    Formel:

    nu          = nu_zaehler/nu_nenner
    f           = (0.79 * np.log(re) - 1.64)**(-2)
    nu_zaehler  = (f/8)*(re-1000.)*pr
    nu_nenner   = 1+12.7*(((f/8)**(1/2))*(pr**(2/3)-1))


    Gültigkeit, turbulent:

    - Prandtl-Zahl:  0.5 <= pr <= 2000
    - Reynolds-Zahl: 3000 <= re <= 5*10**6


    Quelle:

    Gnielinski nach Incropera De Witt 8.63 & 8.21,  S. 492,
    Fundamentals of Heat and Mass Transfer 5th Ed.


    """

    if (pr < 0.7 or (pr > 160)):

        if warnung == 0:
            print("Warnung! Pandtl-Zahl außerhalb Gültigkeit")

    if re < 3000:
        nu = 3.66

    elif re < 5e6:
        f = (0.79 * np.log(re) - 1.64) ** (-2)
        nu_zaehler = (f / 8) * (re - 1000.) * pr
        nu_nenner = 1 + 12.7 * (((f / 8) ** (1 / 2)) * (pr ** (2 / 3) - 1))
        nu = nu_zaehler / nu_nenner

    else:

        if warnung == 0:
            print("Warnung! Reynoldszahl zu groß")

        nu = 0

    return nu


nusselt_zahl_1 = np.vectorize(nusselt_zahl_1)  # Unnötig da durch "np.log()"


# Funktion bereits bekannt für numpy
###############################################################################


def nusselt_zahl_2(re: float, pr: float, n=0.4):
    """
    Berechnung der Nusselt-Zahl nach Korrelation 2


    Variablen:

    - re : Reynolds-Zahl [-]
    - pr : Prandtl-Zahl [-]
    - nu : Nusselt-Zahl [-]
    - n  : Exponent
              - heizen = 0.4
              - kühlen = 0.3

    Formel:

    nu = 0.023*(re**(0.8))*(pr**n)


    Gültigkeit, turbulent:

    - Prandtl-Zahl:  0.6 <= pr <= 160
    - Reynolds-Zahl: 10000 <= re


    Quelle:

    Dittus-Boelter-Korrelation
    Incropera De Witt 8.60,  S. 491,
    Fundamentals of Heat and Mass Transfer 5th Ed.


    """

    if (pr < 0.7 or (pr > 160)):

        if warnung == 0:
            print("Warnung! Pandtl-Zahl außerhalb Gültigkeit")

    if re < 3000:
        nu = 3.66

    elif re >= 1e4:
        nu = 0.023 * (re ** (0.8)) * (pr ** n)

    else:

        if warnung == 0:
            print("Warnung! Reynolds-Zahl im Übergangsbereich")

        nu = 3.66  # nicht korrekt!

    return nu


nusselt_zahl_2 = np.vectorize(nusselt_zahl_2)


###############################################################################


def waermeuebergang_koeff(nu: float, lambd: float, d: float):
    """
    Berechnung des Wärmeübergangskoeffizienten


    Variablen:

    - nu    : Nusselt-Zahl, [-]
    - d     : Durchmesser [m]
    - lambd : Wärmeleitfähigkeit [W/m K]
    - alpha : Wärmeübergangskoeffizient [W/m²K]


    Formel:

    a = (nu*lambd)/d


    """

    alpha = (nu * lambd) / d

    return alpha


waermeuebergang_koeff = np.vectorize(waermeuebergang_koeff)


###############################################################################


def berechnung_re_nu_alpha(u: float, d: float, v: float, pr: float, \
                           lambd: float, korrelation: int):
    """
    Berechnung:

    - Reynolds-Zahl in Abhängigkeit zur Geschwindigkeit des betrachteten
      Fluides sowie des Durchmessers des durchströmten Rohres
    - Nusselt-Zahl in Abhängigkeit zur Reynolds-Zahl
    - Wärmeübergangskoeffizient in Abhängigkeit zur Nusselt-Zahl und der
      Wärmeleitfähigkeit


    Variablen:

    - u : Geschwindigkeit [m/s]
    - d : Durchmesser [m]
    - v : Kinematische Viskosität [m²/s]
    - re: Reynolds-Zahl [-]
    - pr: Prandtl-Zahl [-]
    - nu: Nusselt-Zahl [-]

    - lambd : Wärmeleitfähigkeit [W/m K]
    - alpha : Wärmeübergangskoeffizient [W/m²K]

    - korrerlation : Auswahl der Nusselt-Korrelation


    """

    re = reynolds_zahl(u, d, v)

    if korrelation == 1:
        nu = nusselt_zahl_1(re, pr)

    else:
        nu = nusselt_zahl_2(re, pr)

    alpha = waermeuebergang_koeff(nu, lambd, d)

    return re, nu, alpha


berechnung_re_nu_alpha = np.vectorize(berechnung_re_nu_alpha)

### Skript  ###

###############################################################################


if __name__ == "__main__":

    """
    Der Befehl

    " if __name__ == "__main__": " 

    ermöglicht den import von Funktionen aus dieser Datei in ein weiteres
    Programm und verhindert dabei, dass das ihm folgende Skript dort ebenfalls 
    ausgeführt wird.


    """

    ### Skript - Warnmeldungen ###

    warnung = 1  # Wenn 0, Warnmeldung innerhalb der Konsole

    ### Skript - Variablen - Aufgabe 1a ###

    Fluide = ["Luft", "Wasser", "Ethylenglykol"]
    Color = ["k", "b", "g"]
    Prandtl = np.array([0.707, 5.830, 151.0])  # Prandtl-Zahl [-]
    Re_range = np.geomspace(10, 2e5)  # Reynolds-Zahl [-]

    ### Skript - Berechnung und Plots - Aufgabe 1a ###

    print("Berechnung: Aufgabe 1a")
    print()  # Leerzeile

    for i, Pr in enumerate(Prandtl):
        print("Fluid:", Fluide[i], ", Pr =", Pr)

        ### Berechnung ###

        Nu1 = nusselt_zahl_1(Re_range, Pr)
        Nu2 = nusselt_zahl_2(Re_range, Pr)
        Nu_error = (abs(Nu2[-1] - Nu1[-1])) / Nu1[-1]  # Relativer Fehler

        print("max. Relativer Fehler =", np.round(Nu_error * 100, 2), "%")
        print()

        ### Plot ###

        plt.figure(1)

        plt.plot(Re_range, Nu1, "." + Color[i], label="K, " + str(Pr))
        plt.plot(Re_range, Nu2, "-" + Color[i], )
        plt.legend()
        plt.ylabel(r"$Nu$")
        plt.xlabel(r"$Re$")
        plt.show()

    ### Skript - Variablen - Aufgabe 1b ###

    d = 0.03  # Durchmesser [m]
    u = np.geomspace(0.01, 100)  # Geschwindigkeit [m/s]
    lambd = np.array([26.3e-3, 613e-3, 252e-3])  # Wärmeleitfähigkeit [W/m K]
    v = np.array([15.89e-6, 8.5757e-7, 14.1e-6])  # Kin. Viskosität [m²/s]

    ### Skript - Berechnung und Plots - Aufgabe 1b ###

    print("Berechnung: Aufgabe 1b")
    print()  # Leerzeile

    for i, Pr in enumerate(Prandtl):
        ### Berechnung ###

        Re1, Nu1, alpha1 = berechnung_re_nu_alpha(u, d, v[i], Pr, lambd[i], 1)
        Re2, Nu2, alpha2 = berechnung_re_nu_alpha(u, d, v[i], Pr, lambd[i], 0)

        ### Plot ###

        plt.figure(2)

        plt.plot(u, alpha1, "." + Color[i], label="K, " + str(Pr))
        plt.plot(u, alpha2, "-" + Color[i], )
        plt.legend()
        plt.ylabel(r"$alpha[W/m K]$")
        plt.xlabel(r"$u[m/s]$")
        #plt.show()

    print("Berechnung wurde durchgeführt")