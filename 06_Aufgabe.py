"""
Dienstag, 12.11.24
@author Janik Focken
--------------------------------
Aufgabe 6 - Wasserkocher
Zwei Systeme definiert Heizelement und das Wasser.
"""
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Konstanten für System 1 (Heizelement)
alpha = 800  # Wärmeübergangskoeffizient (W/m²K)
A_h = np.pi * 0.009 * 0.6  # Oberfläche des Heizelements (Durchmesser 9 mm, Länge 60 cm)
c_v_h = 490  # Spezifische Wärmekapazität des Heizelements (J/kgK, Stahl)
dichte_stahl = 7850  # Dichte von Stahl in kg/m³
volumen_heizelement = np.pi * (0.009 / 2)**2 * 0.6  # Volumens des Heizelements
masse_heizelement = volumen_heizelement * dichte_stahl  # Masse Heizelement

# Konstanten für System 2 (Wasser)
volumen_w = 0.001  # Volumen in Kubikmetern (1 Liter)
durchmesser_w = 0.12  # Durchmesser in Metern (12 cm)
hoehe_w = volumen_w / (np.pi * (durchmesser_w / 2)**2)
A_aussen = np.pi * durchmesser_w * hoehe_w + 2 * (np.pi * (durchmesser_w / 2)**2)
alpha_aussen = 10  # Aussen anliegender Wärmeübergangskoeffizient (W/m²K)
epsilon = 0.9  # Emissionsgrad der Außenwand
sigma = 5.67e-8  # Stefan-Boltzmann-Konstante (W/m² K⁴)
T_u = 298  # Umgebungstemperatur (K)
m_w = 1  # Masse des Wassers (kg, 1 Liter)
c_v_w = 4200  # Spezifische Wärmekapazität von Wasser (J/kg K)
