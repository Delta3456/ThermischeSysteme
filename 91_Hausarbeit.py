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
    "Spezifische Enthalpie (h, 10³ J/kg)": [63.078, 142.55],
    "Spezifische Entropie (s, 10³ J/kg·K)": [0.22446, 0.49155],
    "Spezifische isobare Wärmekapazität (cp, 10³ J/kg·K)": [4.189, 4.179],
    "Isobarer Volumen-Ausdehnungskoeffizient (αv, 10⁻³ K⁻¹)": [0.1509, 0.3371],
    "Wärmeleitfähigkeit (kappa, 10⁻³ W/m·K)": [588.8, 620.29],
    "Dynamische Viskosität (eta, 10⁻⁶ Pa·s)": [1137.6, 733.73],
    "Kinematische Viskosität (my, 10⁻⁶ m²/s)": [1.139, 0.7379],
    "Temperaturleitfähigkeit (a, 10⁻⁶ m²/s)": [0.1407, 0.1493],
    "Prandtl-Zahl (Pr)": [8.093, 4.943],
}

# Stoffwerte Stickstoff p=1bar aus VDI Wärmeatlas
data_stickstoff = {
    "Temperatur (t, °C)": [250, 300, 350, 400, 450, 500],
    "Dichte (rho, kg/m³)": [0.64376, 0.5876, 0.54045, 0.50031, 0.46572, 0.43561],
    "Spezifische Enthalpie (h, 10³ J/kg)": [545, 598.2, 651.9, 706.2, 761.1, 816.6],
    "Spezifische Entropie (s, 10³ J/kg·K)": [7.428, 7.525, 7.615, 7.699, 7.777, 7.851],
    "Spezifische isobare Wärmekapazität (cp, 10³ J/kg·K)": [1.06, 1.07, 1.08, 1.092, 1.104, 1.116],
    "Spezifische isochore Wärmekapazität (cv, 10³ J/kg·K)": [0.763, 0.772, 0.783, 0.795, 0.807, 0.819],
    "Schallgeschwindigkeit (ws, m/s)": [464.7, 485.5, 505.3, 524.1, 542.1, 559.4],
    "Wärmeleitfähigkeit (kappa, 10⁻³ W/m·K)": [40.42, 43.32, 46.13, 48.87, 51.53, 54.14],
    "Dynamische Viskosität (eta, 10⁻⁶ Pa·s)": [26.9, 28.66, 30.35, 31.98, 33.56, 35.08],
    "Kinematische Viskosität (my, 10⁻⁷ m²/s)": [417.9, 487.8, 561.6, 639.2, 720.5, 805.4],
    "Temperaturleitfähigkeit (a, 10⁻⁷ m²/s)": [592.2, 689.3, 790.2, 894.7, 1002.6, 1113.8],
    "Prandtl-Zahl (Pr)": [0.7057, 0.7076, 0.7107, 0.7144, 0.7187, 0.7231],
}

# Kurzbezeichnungen für die Stoffwerte
label_mapping = {
    'Temperatur (t, °C)': 't',
    'Dichte (rho, kg/m³)': 'rho',
    'Spezifische isobare Wärmekapazität (cp, 10³ J/kg·K)': 'cp',
    'Spezifische isochore Wärmekapazität (cv, 10³ J/kg·K)': 'cv',
    'Wärmeleitfähigkeit (kappa, 10⁻³ W/m·K)': 'kappa',
    'Kinematische Viskosität (my, 10⁻⁶ m²/s)': 'my',
    'Kinematische Viskosität (my, 10⁻⁷ m²/s)': 'my',
    'Prandtl-Zahl (Pr)': 'pr',
}

# Faktor um alle Einheiten ohne Probleme nutzen zu können
units_mapping = {
    'Spezifische isobare Wärmekapazität (cp, 10³ J/kg·K)': 10**3,
    'Spezifische isochore Wärmekapazität (cv, 10³ J/kg·K)': 10**3,
    'Wärmeleitfähigkeit (kappa, 10⁻³ W/m·K)': 10**-3,
    'Kinematische Viskosität (my, 10⁻⁶ m²/s)': 10**-6,
    'Kinematische Viskosität (my, 10⁻⁷ m²/s)': 10 **-7,
}


# Funktion um Daten zu konvertieren und mit Labels zu versehen
def convert_data(data, unit_mapping, label_mapping):
    converted_data = {}
    for label, values in data.items():
        factor = unit_mapping.get(label, 1)  # Standard: Keine Umrechnung
        converted_values = [value * factor for value in values]
        # Temperaturen in Kelvin umrechnen, falls das Label Temperatur enthält
        if 'Temperatur' in label:
            converted_values = [value + 273.15 for value in converted_values]
            # Label anpassen
            new_label = label.replace('(t, °C)', '(t, K)')
            converted_data[new_label] = converted_values
            # Kurzbezeichnung anpassen
            if label in label_mapping:
                short_key = label_mapping[label]
                converted_data[short_key] = converted_values
        else:
            converted_data[label] = converted_values
            if label in label_mapping:
                short_key = label_mapping[label]
                converted_data[short_key] = converted_values

        # Kurzbezeichnung hinzufügen, falls verfügbar
        if label in label_mapping:
            short_key = label_mapping[label]
            converted_data[short_key] = converted_values
    return converted_data


# Funktion zur Interpolation
def interpolate_value(data, x_label, y_label, x_value):
    """
    Interpoliert einen Wert basierend auf gegebenen Daten.
    Beispiel für Cp von Wasser bei 25°C:
    cp_25 = interpolate_value(data_wasser_converted, 't','cp', 25)
    """
    x_data = data[x_label]  # Werte der unabhängigen Variablen
    y_data = data[y_label]  # Werte der abhängigen Variablen

    # Überprüfen, ob der Wert im Bereich der Daten liegt
    if not (min(x_data) <= x_value <= max(x_data)):
        raise ValueError(f"{x_value} liegt außerhalb des Bereichs {min(x_data)} bis {max(x_data)}")

    # Lineare Interpolation
    return np.interp(x_value, x_data, y_data)


# Daten konvertieren
data_stickstoff_converted = convert_data(data_stickstoff, units_mapping, label_mapping)
data_wasser_converted = convert_data(data_wasser, units_mapping, label_mapping)
# Test print('cp', data_wasser_converted["cp"])

# Werte für Rohr
m_dot_w = 0.2  # kg/s
T_w_ein = 15 + 273.15  # K
T_w_aus = 35 + 273.15  # K
dT_erf = T_w_aus - T_w_ein  # delta t was erforderlich ist um das Wasser aufzuwärmen
T_w_m = (T_w_ein + T_w_aus) / 2  # Mittlere Temperatur des Wassers
cp_w_Tm = interpolate_value(data_wasser_converted, 't', 'cp', T_w_m)
rho_w_Tm = interpolate_value(data_wasser_converted, 't', 'rho', T_w_m)
my_w_Tm = interpolate_value(data_wasser_converted, 't', 'my', T_w_m)
Pr_w_Tm = interpolate_value(data_wasser_converted, 't', 'pr', T_w_m)
kappa_w_Tm = interpolate_value(data_wasser_converted, 't', 'kappa', T_w_m)
# Erforderlicher Wärmestrom der nötig ist um das Wasser aufzuwärmen
# Annahme, dass die bei flüssigem Wasser sich nicht stark ändert, daher wird der Mittelwert genutzt
Q_dot_erf = m_dot_w * dT_erf * cp_w_Tm  # Watt

# Werte zum varieren
# Durchmesser- und Längenbereiche für das Rohr
D_r_Werte = np.linspace(0.02, 0.04, 5)  # Meter
L_r_Werte = np.linspace(3, 6, 4)  # Meter

# Gasgeschwindigkeit und Temperaturbereiche für Abgas
u_ab_Werte = np.linspace(20, 40, 5)  # m/s
T_ab_Werte = np.linspace(250 + 273.15, 500 + 273.15, 6)  # K


# Thermodynamische Funktionen -----------------------------------------------------------
def berechne_re(rho, u, D, my):
    """Berechnet die Reynolds-Zahl."""
    return (rho * u * D) / my


def berechne_nu_rohr(Re, Pr):
    """Berechnet die Nusselt-Zahl für turbulente Strömung in Rohren (Dittus-Boelter-Gleichung)."""
    return 0.023 * Re ** 0.8 * Pr ** 0.4

def berechne_nu_ab(Re, Pr):
    """Berechnet die Nusselt-Zahl für Strömung (Gnielinski-Gleichung)."""
    if Re < 2300:
        Nu = 3.66  # Laminare Strömung
    elif Re <= 5e6:
        f = (0.79 * np.log(Re) - 1.64) ** -2
        Nu = (f / 8) * (Re - 1000) * Pr / (1 + 12.7 * (f / 8) ** 0.5 * (Pr ** (2/3) - 1))
    else:
        Nu = np.nan  # Außerhalb des Gültigkeitsbereichs
    return Nu


def berechne_waermeuebergangsko(Nu, kappa, D):
    """Berechnet den Wärmeübergangskoeffizienten."""
    return Nu * kappa / D


def berechne_dT_lm(T_hot_in, T_hot_out, T_cold_in, T_cold_out):
    """Berechnet die logarithmische mittlere Temperaturdifferenz."""
    delta_T1 = T_hot_in - T_cold_out
    delta_T2 = T_hot_out - T_cold_in
    if delta_T1 == delta_T2:
        return delta_T1  # Vermeidung von Division durch null
    else:
        return (delta_T1 - delta_T2) / np.log(delta_T1 / delta_T2)


def berechne_waermeuebergangsko_ges(alpha_i, alpha_o):
    """Berechnet den Gesamtwärmeübergangskoeffizienten alpha_ge."""
    return 1 / (1 / alpha_i + 1 / alpha_o)


def berechne_waermestrom(alpha_ges, A, dT_lm):
    """Berechnet den Wärmestrom Q_dot."""
    return alpha_ges * A * dT_lm


# Hauptberechnung -----------------------
# Ergebnisse speichern
ergebnisse = []

# Schleife über alle Kombinationen
for D_r in D_r_Werte:
    for L_r in L_r_Werte:
        A_rohr = np.pi * (D_r / 2) ** 2  # Querschnittsfläche des Rohrs
        u_w = m_dot_w / (rho_w_Tm * A_rohr)  # Wassergeschwindigkeit im Rohr
        Re_w = berechne_re(rho_w_Tm, u_w, D_r, my_w_Tm)
        Nu_w = berechne_nu_rohr(Re_w, Pr_w_Tm)
        alpha_w = berechne_waermeuebergangsko(Nu_w, kappa_w_Tm, D_r)

        for u_ab in u_ab_Werte:
            for T_ab in T_ab_Werte:
                A_wu = np.pi * D_r * L_r  # Wärmeübertragerfläche
                # Annahme, dass T_ab keine Temperaturänderung erfährt
                dT_lm = berechne_dT_lm(T_ab, T_ab, T_w_ein, T_w_aus)

                rho_ab = interpolate_value(data_stickstoff_converted, 't','rho', T_ab)
                my_ab = interpolate_value(data_stickstoff_converted, 't','my', T_ab)
                Re_ab = berechne_re(rho_ab, u_ab, D_r,my_ab)
                Pr_ab = interpolate_value(data_stickstoff_converted, 't','pr', T_ab)
                Nu_ab = berechne_nu_ab(Re_ab, Pr_ab)
                kappa_ab = interpolate_value(data_stickstoff_converted, 't','kappa', T_ab)
                alpha_ab = berechne_waermeuebergangsko(Nu_ab, kappa_ab, D_r)
                alpha_ges = berechne_waermeuebergangsko_ges(alpha_w, alpha_ab)
                Q_dot = berechne_waermestrom(alpha_ges, A_wu, dT_lm)

                #print(f"D_r: {D_r}, L_r: {L_r}, u_ab: {u_ab}, T_ab: {T_ab}, dt_lmP: {dT_lm}: Q_dot: {Q_dot}")
                # Ergebnisse speichern
                ergebnisse.append({
                    'D_r': D_r,
                    'L_r': L_r,
                    'u_ab': u_ab,
                    'T_ab': T_ab,
                    'Q_dot': Q_dot,
                    'Erfuellt Anforderung': Q_dot >= Q_dot_erf
                })
                #print(ergebnisse)

# Ergebnisse in ein DataFrame konvertieren
df = pd.DataFrame(ergebnisse)

# Plotten der Ergebnisse ---------------------------------------------
# Einfluss von D und L auf Q_dot
plt.figure(figsize=(8, 6))
for L_r in L_r_Werte:
    subset = df[
        (df['L_r'] == L_r) &
        (df['u_ab'] == u_ab_Werte[-1]) &
        (df['T_ab'] == T_ab_Werte[-1])
        ]
    plt.plot(subset['D_r'], subset['Q_dot'], label=f'L = {L_r} m')

plt.axhline(y=Q_dot_erf, color='r', linestyle='--', label='Erforderlicher Q̇')
plt.xlabel('Rohrdurchmesser D (m)')
plt.ylabel('Wärmestrom Q̇ (W)')
plt.title('Einfluss des Rohrdurchmessers auf den Wärmestrom')
plt.legend()
plt.grid(True)
plt.show()

# Einfluss von u_gas und T_gas auf Q_dot
plt.figure(figsize=(8, 6))
for T_ab in T_ab_Werte:
    subset = df[
        (df['D_r'] == D_r_Werte[-1]) &
        (df['L_r'] == L_r_Werte[-1]) &
        (df['T_ab'] == T_ab)
        ]
    plt.plot(subset['u_ab'], subset['Q_dot'], label=f'T_gas = {T_ab:.1f} K')

plt.axhline(y=Q_dot_erf, color='r', linestyle='--', label='Erforderlicher Q̇')
plt.xlabel('Gasgeschwindigkeit u_gas (m/s)')
plt.ylabel('Wärmestrom Q̇ (W)')
plt.title('Einfluss der Gasgeschwindigkeit auf den Wärmestrom')
plt.legend()
plt.grid(True)
plt.show()

# Kombinationen, die die Anforderung erfüllen
erfuellt = df[df['Erfuellt Anforderung']]

print("Kombinationen, die den erforderlichen Wärmestrom erfüllen:")
print(erfuellt[['D_r', 'L_r', 'u_ab', 'T_ab']])