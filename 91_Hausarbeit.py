"""
Montag, 02.12.24
@author Janik Focken
--------------------------------
Erste Hausarbeit - Analyse der Wärmeübertragung in einem Rohr
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Stoffwerte Wasser bei p=1bar aus VDI Wärmeatlas
data_wasser = {
    "Temperatur (t, °C)": [15, 35],
    "Dichte (rho, kg/m³)": [999.1, 994.38],
    "Spezifische Enthalpie (h, 10³J/kg)": [63.078, 142.55],
    "Spezifische Entropie (s, 10³J/kg·K)": [0.22446, 0.49155],
    "Spezifische isobare Wärmekapazität (cp, 10³J/kg·K)": [4.189, 4.179],
    "Isobarer Volumen-Ausdehnungskoeffizient (αv, 10⁻³ K⁻¹)": [0.1509, 0.3371],
    "Wärmeleitfähigkeit (lambda, 10⁻³ W/m·K)": [588.8, 620.29],
    "Dynamische Viskosität (eta, 10⁻⁶ Pa·s)": [1137.6, 733.73],
    "Kinematische Viskosität (my, 10⁻⁶ m²/s)": [1.139, 0.7379],
    "Temperaturleitfähigkeit (a, 10⁻⁶ m²/s)": [0.1407, 0.1493],
    "Prandtl-Zahl (Pr)": [8.093, 4.943],
}

# Stoffwerte Stickstoff p=1bar aus VDI Wärmeatlas
data_stickstoff = {
    "Temperatur (t, °C)": [250, 300, 350, 400, 450, 500],
    "Dichte (rho, kg/m³)": [0.64376, 0.5876, 0.54045, 0.50031, 0.46572, 0.43561],
    "Spezifische Enthalpie (h, 10³J/kg)": [545, 598.2, 651.9, 706.2, 761.1, 816.6],
    "Spezifische Entropie (s, 10³J/kg·K)": [7.428, 7.525, 7.615, 7.699, 7.777, 7.851],
    "Spezifische isobare Wärmekapazität (cp, 10³J/kg·K)": [1.06, 1.07, 1.08, 1.092, 1.104, 1.116],
    "Spezifische isochore Wärmekapazität (cv, 10³J/kg·K)": [0.763, 0.772, 0.783, 0.795, 0.807, 0.819],
    "Schallgeschwindigkeit (ws, m/s)": [464.7, 485.5, 505.3, 524.1, 542.1, 559.4],
    "Wärmeleitfähigkeit (lambda, 10⁻³ W/m·K)": [40.42, 43.32, 46.13, 48.87, 51.53, 54.14],
    "Dynamische Viskosität (eta, 10⁻⁶ Pa·s)": [26.9, 28.66, 30.35, 31.98, 33.56, 35.08],
    "Kinematische Viskosität (my, 10⁻⁷ m²/s)": [417.9, 487.8, 561.6, 639.2, 720.5, 805.4],
    "Temperaturleitfähigkeit (a, 10⁻⁷ m²/s)": [592.2, 689.3, 790.2, 894.7, 1002.6, 1113.8],
    "Prandtl-Zahl (Pr)": [0.7057, 0.7076, 0.7107, 0.7144, 0.7187, 0.7231],
}

# neue Label für Variablennamen
label_for_var = {
    'Temperatur (t, °C)': 't',
    'Dichte (rho, kg/m³)': 'rho',
    'Spezifische isobare Wärmekapazität (cp, 10³J/kg·K)': 'cp',
    'Spezifische isochore Wärmekapazität (cv, 10³J/kg·K)': 'cv',
    'Wärmeleitfähigkeit (lambda, 10⁻³ W/m·K)': 'lambda',
    'Prandtl-Zahl (Pr)': 'pr',
}

# Interpolieren und Daten in auf gleiche Einheiten bringen
# Fehlt noch
