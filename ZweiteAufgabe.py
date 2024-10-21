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
def berechne_temperaturverlauf(material, alpha, t_b, t_u, l1, u, a, adiabat=False):
    lam = stoffwerte[material]["lambda_300"]
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

# Plotten Teilaufgabe 1 und 2 --------------------
alpha = 5  # Wärmeübergangskoeffizient (W/m²K)
t_b = 370  # Temperatur der Grundfläche (K)
t_u = 290  # Umgebungstemperatur (K)
l1 = 0.02  # Länge der Rippe (m)
b1 = 0.04  # Breite1 der Grundfläche der Rippe (m)
b2 = 0.002  # Breite2 der Grundfläche der Rippe (m)

u = 2 * (b1 + b2)  # Umfang der Rippe (m)
a = b1 * b2  # Fläche der Rippe (m²)

# Temperaturverlauf für Aluminium mit freier Konvektion an der Spitze
x_al, T_al = berechne_temperaturverlauf("aluminium", alpha, t_b, t_u, l1, u, a)
plt.plot(x_al, T_al, label='Aluminium (freie Konvektion)')

# Temperaturverlauf für Stahl mit freier Konvektion an der Spitze
x_st, T_st = berechne_temperaturverlauf("stahl", alpha, t_b, t_u, l1, u, a)
plt.plot(x_st, T_st, label='Stahl (freie Konvektion)')

# Temperaturverlauf für Aluminium mit adiabatischer Spitze
x_al_ad, T_al_ad = berechne_temperaturverlauf("aluminium", alpha, t_b, t_u, l1, u, a, adiabat=True)
plt.plot(x_al_ad, T_al_ad, label='Aluminium (adiabatische Spitze)', linestyle='--')

# Temperaturverlauf für Stahl mit adiabatischer Spitze
x_st_ad, T_st_ad = berechne_temperaturverlauf("stahl", alpha, t_b, t_u, l1, u, a, adiabat=True)
plt.plot(x_st_ad, T_st_ad, label='Stahl (adiabatische Spitze)', linestyle='--')

# Plot Einstellungen
plt.xlabel('Länge der Rippe (m)')
plt.ylabel('Temperatur (K)')
plt.title('Temperaturverlauf')
plt.grid(True)
plt.legend()
plt.show()
