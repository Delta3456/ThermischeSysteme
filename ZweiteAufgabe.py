"""
Mittwoch, 15.10.24
@author Janik Focken
--------------------------------
Aufgabe 2 - Temperaturverlauf Rippe
1/2) freie Konvektion/adiabat, einer Rippe(rechteckig)
"""
import numpy as np
import matplotlib.pyplot as plt


# Funktionen -----------------------
def temperaturverlauf_rippe(material, alpha, t_b, t_u, l1, u, a, adiabat=False):
    """
    Berechnet den Temperaturverlauf einer Rippe mit den Parametern:
    material = Welches Material
    alpha: Wärmeübergangskoeffizient (W/m²K)
    t_b: Temperatur der Grundfläche (K)
    t_u: Temperatur der Umgebung
    l1: Höhe der Rippe
    u: Umfang der Rippe (m)
    a: Fläche der Rippe (m²)
    """
    # Prüfen, ob das Material existiert
    try:
        lam = stoffwerte[material]["lambda_300"]
    except KeyError:
        raise ValueError(f"Material '{material}' ist nicht definiert.")

    m = np.sqrt((alpha * u) / (lam * a))
    x = np.linspace(0, l1, 100)
    theta_b = t_b - t_u

    if adiabat:
        # Adiabatische Spitze: keine Wärmeabgabe an der Spitze
        theta = (
                    np.cosh(m * (l1 - x))
                ) / (
                    np.cosh(m * l1)
                )
    else:
        # Freie Konvektion an der Spitze
        theta = ((np.cosh(m * (l1 - x)) + (alpha / (m * lam)) * np.sinh(m * (l1 - x)))
                 /
                 (np.cosh(m * l1) + (alpha / (m * lam)) * np.sinh(m * l1)))

    t = theta * theta_b + t_u
    return x, t


def waermestrom_rippe(material, alpha, t_b, t_u, l1, u, a):
    # Berechnet den Wärmestrom an einer Rippe
    # Prüfen, ob das Material existiert
    try:
        lam = stoffwerte[material]["lambda_300"]
    except KeyError:
        raise ValueError(f"Material '{material}' ist nicht definiert.")
    m = np.sqrt((alpha * u) / (lam * a))
    theta_b = t_b - t_u

    z1 = np.sinh(m * l1) + (alpha / (m * lam)) * np.cosh(m * l1)
    z2 = np.cosh(m * l1) + (alpha / (m * lam)) * np.sinh(m * l1)
    q = m * lam * a * theta_b * (z1 / z2)
    return q


def umfang_rechteck(b1, b2):
    # Berechnet den Umfang eines Rechtecks
    return 2 * (b1 + b2)


def flaeche_rechteck(b1, b2):
    # Berechnet die Fläche eines Rechtecks
    return b1 * b2


# Stoffwerte
stoffwerte = {
    "aluminium": {
        "lambda_300": 177,  # Wärmeleitfähigkeit (W/mK) bei 300 K
        "lambda_600": 186,  # Wärmeleitfähigkeit (W/mK) bei 600 K
        "rho": 2770,  # Dichte (kg/m³)
        "cp": 875  # spezifische Wärmekapazität (J/kg/K) bei 300 K
    },
    "stahl": {
        "lambda_300": 14.9,  # Wärmeleitfähigkeit (W/mK) bei 300 K
        "lambda_600": 19.8,  # Wärmeleitfähigkeit (W/mK) bei 600 K
        "rho": 7900,  # Dichte (kg/m³)
        "cp": 477  # spezifische Wärmekapazität (J/kg/K) bei 300 K
    }
}

# Parameter
alpha = 5  # Wärmeübergangskoeffizient (W/m²K)
t_b = 370  # Temperatur der Grundfläche (K)
t_u = 290  # Umgebungstemperatur (K)
l1 = 0.02  # Länge der Rippe (m)
b1 = 0.04  # Breite1 der Grundfläche der Rippe (m)
b2 = 0.002  # Breite2 der Grundfläche der Rippe (m)

u = umfang_rechteck(b1,b2)  # Umfang der Rippe (m)
a = flaeche_rechteck(b1,b2)  # Fläche der Rippe (m²)

# Teilaufgabe 1 und 2 --------------------
"""Hier könnte zur optimierung eine for-schleife sein"""
# Temperaturverlauf für Aluminium mit freier Konvektion an der Spitze
x_al, T_al = temperaturverlauf_rippe("aluminium", alpha, t_b, t_u, l1, u, a)
plt.plot(x_al, T_al, label='Aluminium (freie Konvektion)')

# Temperaturverlauf für Stahl mit freier Konvektion an der Spitze
x_st, T_st = temperaturverlauf_rippe("stahl", alpha, t_b, t_u, l1, u, a)
plt.plot(x_st, T_st, label='Stahl (freie Konvektion)')

# Temperaturverlauf für Aluminium mit adiabatischer Spitze
x_al_ad, T_al_ad = temperaturverlauf_rippe("aluminium", alpha, t_b, t_u, l1, u, a, adiabat=True)
plt.plot(x_al_ad, T_al_ad, label='Aluminium (adiabatische Spitze)', linestyle='--')

# Temperaturverlauf für Stahl mit adiabatischer Spitze
x_st_ad, T_st_ad = temperaturverlauf_rippe("stahl", alpha, t_b, t_u, l1, u, a, adiabat=True)
plt.plot(x_st_ad, T_st_ad, label='Stahl (adiabatische Spitze)', linestyle='--')

# Plot Einstellungen
plt.xlabel('Länge der Rippe (m)')
plt.ylabel('Temperatur (K)')
plt.title('Temperaturverlauf')
plt.grid(True)
plt.legend()
plt.show()

# Teilaufgabe 3 ------------------------
V_total = l1 * b1 * b2  # Gesamtvolumen der Rippe

rippen_array = [1, 2, 3, 4]  # Anzahl der Rippen
material = "aluminium"  # Materialauswahl

Q_total_hoehe = []
Q_total_breite = []
Q_total_dicke = []

for rippen in rippen_array:
    # Variation der Höhe (Länge)
    l_n = V_total / (rippen * b1 * b2)
    u = umfang_rechteck(b1, b2)
    a = flaeche_rechteck(b1, b2)
    Q_rippe = waermestrom_rippe(material, alpha, t_b, t_u, l_n, u, a)
    Q_total = rippen * Q_rippe
    Q_total_hoehe.append(Q_total)

    # Variation der Breite
    b1_N = V_total / (rippen * l1 * b2)
    u = umfang_rechteck(b1_N, b2)
    a = flaeche_rechteck(b1_N, b2)
    Q_rippe = waermestrom_rippe(material, alpha, t_b, t_u, l1, u, a)
    Q_total = rippen * Q_rippe
    Q_total_breite.append(Q_total)

    # Variation der Dicke
    b2_N = V_total / (rippen * l1 * b1)
    u = umfang_rechteck(b1, b2_N)
    a = flaeche_rechteck(b1, b2_N)
    Q_rippe = waermestrom_rippe(material, alpha, t_b, t_u, l1, u, a)
    Q_total = rippen * Q_rippe
    Q_total_dicke.append(Q_total)

# Ergebnisse plotten
plt.figure(figsize=(10, 6))
plt.plot(rippen_array, Q_total_hoehe, linestyle='-', label='Variation der Höhe')
plt.plot(rippen_array, Q_total_breite, linestyle='--', label='Variation der Breite')
plt.plot(rippen_array, Q_total_dicke, linestyle='-.', label='Variation der Dicke')
plt.xlabel('Anzahl der Rippen (N)')
plt.ylabel('Gesamtwärmestrom Q (W)')
plt.title('Gesamtwärmestrom in Abhängigkeit von der Anzahl der Rippen')
plt.grid(True)
plt.legend()
plt.show()