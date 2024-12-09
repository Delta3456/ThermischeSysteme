"""
Montag, 02.12.24
@author Janik Focken
--------------------------------
Erste Hausarbeit - Analyse der Wärmeübertragung in einem Rohr
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

# Stoffwerte Wasser bei p=1bar aus VDI Wärmeatlas(Kapitel C2.1)
data_wasser = {
    "Temperatur (t, °C)": [15, 34],
    "Dichte (rho, kg/m³)": [999.1, 994.38],
    "Spezifische Enthalpie (h, 10³ J/kg)": [63.078, 142.55],
    "Spezifische Entropie (s, 10³ J/kg·K)": [0.22446, 0.49155],
    "Spezifische isobare Wärmekapazität (cp, 10³ J/kg·K)": [4.189, 4.179],
    "Isobarer Volumen-Ausdehnungskoeffizient (αv, 10⁻³ K⁻¹)": [0.1509, 0.3371],
    "Wärmeleitfähigkeit (kappa, 10⁻³ W/m·K)": [588.8, 620.29],
    "Dynamische Viskosität (eta, 10⁻⁶ Pa·s)": [1137.6, 733.73],
    "Kinematische Viskosität (ny, 10⁻⁶ m²/s)": [1.139, 0.7379],
    "Temperaturleitfähigkeit (a, 10⁻⁶ m²/s)": [0.1407, 0.1493],
    "Prandtl-Zahl (Pr)": [8.093, 4.943],
}

# Stoffwerte trockene Luft p=1bar aus VDI Wärmeatlas(Kapitel C2.3)
data_luft = {
    "Temperatur (t, °C)": [250, 300, 350, 400, 450, 500],
    "Dichte (rho, kg/m³)": [0.6655, 0.6075, 0.5587, 0.5172, 0.4815, 0.4503],
    "Spezifische Enthalpie (h, 10³ J/kg)": [228.9, 280.9, 333.5, 386.6, 440.3, 494.7],
    "Spezifische Entropie (s, 10³ J/kg·K)": [0.5751, 0.6700, 0.7579, 0.8399, 0.9169, 0.9896],
    "Spezifische isobare Wärmekapazität (cp, 10³ J/kg·K)": [1.035, 1.045, 1.057, 1.069, 1.081, 1.093],
    "Spezifische isochore Wärmekapazität (cv, 10³ J/kg·K)": [0.7472, 0.7580, 0.7695, 0.7815, 0.7935, 0.8054],
    "Schallgeschwindigkeit (ws, m/s)": [456.2, 476.6, 495.9, 514.3, 532.0, 549.0],
    "Wärmeleitfähigkeit (kappa, 10⁻³ W/m·K)": [41.38, 44.42, 47.37, 50.24, 53.05, 55.80],
    "Dynamische Viskosität (eta, 10⁻⁶ Pa·s)": [27.97, 29.81, 31.58, 33.28, 34.93, 36.53],
    "Kinematische Viskosität (ny, 10⁻⁶ m²/s)": [4.203, 4.907, 5.652, 6.435, 7.256, 8.112],
    "Temperaturleitfähigkeit (a, 10⁻⁷ m²/s)": [601.0, 699.5, 802.2, 908.9, 1019.5, 1133.9],
    "Prandtl-Zahl (Pr)": [0.6993, 0.7016, 0.7046, 0.7080, 0.7117, 0.7154],
}

# Kurzbezeichnungen für die Stoffwerte
label_mapping = {
    'Temperatur (t, °C)': 't',
    'Dichte (rho, kg/m³)': 'rho',
    'Spezifische isobare Wärmekapazität (cp, 10³ J/kg·K)': 'cp',
    'Spezifische isochore Wärmekapazität (cv, 10³ J/kg·K)': 'cv',
    'Wärmeleitfähigkeit (kappa, 10⁻³ W/m·K)': 'kappa',
    'Kinematische Viskosität (ny, 10⁻⁶ m²/s)': 'ny',
    'Prandtl-Zahl (Pr)': 'pr',
}

# Faktor um alle Einheiten ohne Probleme nutzen zu können
units_mapping = {
    'Spezifische isobare Wärmekapazität (cp, 10³ J/kg·K)': 10 ** 3,
    'Spezifische isochore Wärmekapazität (cv, 10³ J/kg·K)': 10 ** 3,
    'Wärmeleitfähigkeit (kappa, 10⁻³ W/m·K)': 10 ** -3,
    'Kinematische Viskosität (ny, 10⁻⁶ m²/s)': 10 ** -6,
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
    Beispiel für Cp von Wasser bei 25 °C:
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
data_luft_converted = convert_data(data_luft, units_mapping, label_mapping)
data_wasser_converted = convert_data(data_wasser, units_mapping, label_mapping)

# Werte für Rohr
m_dot_w = 0.2  # kg/s
T_w_ein = 15 + 273.15  # K
T_w_aus = 35 + 273.15  # K
dT_erf = T_w_aus - T_w_ein  # delta t was erforderlich ist um das Wasser aufzuwärmen
T_w_m = (T_w_ein + T_w_aus) / 2  # Mittlere Temperatur des Wassers
# Stoffwerte von Wasser bei mittlerer Temperatur ist notwenig für die Berechnung weiterer Werte z.B. Nusselt nach VDI(G6, S.840)
cp_w_Tm = interpolate_value(data_wasser_converted, 't', 'cp', T_w_m)
rho_w_Tm = interpolate_value(data_wasser_converted, 't', 'rho', T_w_m)
ny_w_Tm = interpolate_value(data_wasser_converted, 't', 'ny', T_w_m)
Pr_w_Tm = interpolate_value(data_wasser_converted, 't', 'pr', T_w_m)
kappa_w_Tm = interpolate_value(data_wasser_converted, 't', 'kappa', T_w_m)
# Erforderlicher Wärmestrom der nötig ist um das Wasser aufzuwärmen
# Annahme, dass die bei flüssigem Wasser sich nicht stark ändert, daher wird der Mittelwert genutzt
Q_dot_erf = m_dot_w * dT_erf * cp_w_Tm  # Watt
Q_dot_erf = float(Q_dot_erf)

# Parameterbereiche
# Durchmesser- und Längenbereiche für das Rohr
D_r_Werte = np.linspace(0.02, 0.04, 5)  # Meter
L_r_Werte = np.linspace(3, 6, 4)  # Meter

# Gasgeschwindigkeit und Temperaturbereiche für Abgas
u_ab_Werte = np.linspace(20, 40, 5)  # m/s
T_ab_Werte = np.linspace(250 + 273.15, 500 + 273.15, 6)  # K


# Thermodynamische Funktionen -----------------------------------------------------------
def berechne_re(u, L, ny):
    """Berechnet die Reynolds-Zahl."""
    return (u * L) / ny


def berechne_nu_rohr(Re, Pr, d_rohr, l_rohr):
    """
    Berechnet die Nusselt-Zahl bei voll ausgebildeter turbulenter Strömung(Re <= 4 ** 3)
    nach Gnielinski(VDI-Wärmeatlas, Kaptiel G1, 4.1 Formel 28)
    """
    if Re >= 10 ** 6 or Re <= 4 ** 3:
        raise ValueError("Reynolds-Zahl muss größer 4 ** 3 sein.")
    xi = (1.8 * math.log10(Re) - 1.5) ** -2
    nu_rohr = ((xi / 8) * (Re - 1000) * Pr) / (1 + 12.7 * math.sqrt(xi / 8) * (Pr ** (2 / 3) - 1))
    nu_rohr *= 1 + ((1/3) * (d_rohr / l_rohr)) ** (2 / 3)
    return nu_rohr


def berechne_nu_ab(Re_D, Pr):
    """
    Berechnet die mittlere Nusselt-Zahl für eine Querströmung über einen Zylinder.
    nach Incropera, Frank P.: Fundamentals of heat and mass transfer(S.420. Formel:7.54)
    """
    term1 = (0.62 * (Re_D ** 0.5) * (Pr ** (1 / 3))) / (1 + (0.4 / Pr) ** (2 / 3)) ** 0.25
    term2 = (1 + (Re_D / 282000) ** (5 / 8)) ** (4 / 5)
    Nu = 0.3 + term1 * term2
    return Nu



def berechne_waermeuebergangsko(Nu, kappa, D):
    """Berechnet den Wärmeübergangskoeffizienten."""
    return Nu * kappa / D


def berechne_dT_lm(T_hot_in, T_hot_out, T_cold_in, T_cold_out):
    """Berechnet die logarithmische mittlere Temperaturdifferenz."""
    delta_T1 = T_hot_in - T_cold_out
    delta_T2 = T_hot_out - T_cold_in
    if delta_T1 == delta_T2:
        return 1/2 * (delta_T1 + delta_T2)  # Vermeidung von Division durch null
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
        Re_w = berechne_re(u_w, D_r, ny_w_Tm)
        Nu_w = berechne_nu_rohr(Re_w, Pr_w_Tm, D_r, L_r)
        alpha_w = berechne_waermeuebergangsko(Nu_w, kappa_w_Tm, D_r)

        for u_ab in u_ab_Werte:
            for T_ab in T_ab_Werte:
                A_wu = np.pi * D_r * L_r  # Wärmeübertragerfläche
                # Annahme, dass T_ab keine Temperaturänderung erfährt
                dT_lm = berechne_dT_lm(T_ab, T_ab, T_w_ein, T_w_aus)

                ny_ab = interpolate_value(data_luft_converted, 't', 'ny', T_ab)
                Re_ab = berechne_re(u_ab, D_r, ny_ab)
                Pr_ab = interpolate_value(data_luft_converted, 't', 'pr', T_ab)
                Nu_ab = berechne_nu_ab(Re_ab, Pr_ab)
                kappa_ab = interpolate_value(data_luft_converted, 't', 'kappa', T_ab)
                alpha_ab = berechne_waermeuebergangsko(Nu_ab, kappa_ab, D_r)
                alpha_ges = berechne_waermeuebergangsko_ges(alpha_w, alpha_ab)
                Q_dot = berechne_waermestrom(alpha_ges, A_wu, dT_lm)

                # Ergebnisse speichern
                ergebnisse.append({
                    'D_r': D_r,
                    'L_r': L_r,
                    'u_ab': u_ab,
                    'T_ab': T_ab,
                    'Q_dot': Q_dot,
                    'Erfuellt Anforderung': Q_dot >= Q_dot_erf
                })

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

plt.axhline(y=Q_dot_erf, color='r', linestyle='--', label='Erforderlicher $\dot{Q}$')
plt.xlabel('Rohrdurchmesser D_r (m)')
plt.ylabel('Wärmestrom $\dot{Q}$ (W)')
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

plt.axhline(y=Q_dot_erf, color='r', linestyle='--', label='Erforderlicher $\dot{Q}$')
plt.xlabel('Abgasgeschwindigkeit u (m/s)')
plt.ylabel('Wärmestrom $\dot{Q}$ (W)')
plt.title('Einfluss der Gasgeschwindigkeit auf den Wärmestrom')
plt.legend()
plt.grid(True)
plt.show()

# Kombinationen, die die Anforderung erfüllen
erfuellt = df[df['Erfuellt Anforderung']]

print("Kombinationen, die den erforderlichen Wärmestrom erfüllen:")
print(erfuellt[['D_r', 'L_r', 'u_ab', 'T_ab']])
