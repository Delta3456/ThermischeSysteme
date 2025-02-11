"""
Montag, 10.02.25
@author Janik Focken
--------------------------------
Dritte Hausarbeit - Reaktor
"""

import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.integrate import solve_ivp
from CoolProp.CoolProp import PropsSI
from functools import partial
from dataclasses import dataclass
from typing import Tuple, List

# Globale Konstante zur Umrechnung von °C in Kelvin
CELSIUS_TO_KELVIN: float = 273.15

# ---------------------------------------------------------
# Data Class zur Definition der Reaktorparameter
# ---------------------------------------------------------
@dataclass
class ReactorParameters:
    m_dot: float         # Massenstrom [kg/s]
    cp: float            # spezifische Wärmekapazität [J/(kg·K)]
    alpha_i: float       # innerer Wärmeübergangskoeffizient [W/(m²·K)]
    d_in: float          # Innendurchmesser des Rohrs [m]
    T_wall_reac: float   # (Initialer) Platzhalter für Wandtemperatur [°C] – wird dynamisch überschrieben
    n_dot_meoh: float    # Stoffmengenstrom Methanol [mol/s]
    dH_reac: float       # Reaktionsenthalpie [J/mol]
    k_reac: float        # kinetischer Parameter [1/m]
    q_in_segment: float = 0.0  # Heizleistung pro Meter [W/m] (wird in jedem Segment gesetzt)

# ---------------------------------------------------------
# A) Fluid-Eigenschaften (mittels CoolProp)
# ---------------------------------------------------------
def fluid_properties(T_degC: float, p_pa: float, x_methanol: float = 0.5) -> Tuple[float, float, float]:
    """
    Bestimmt Dichte, spezifische Wärmekapazität und Wärmeleitfähigkeit des Fluids.
    """
    T_k: float = T_degC + CELSIUS_TO_KELVIN  # Umrechnung in Kelvin
    fluid_str: str = f"HEOS::Methanol[{x_methanol}]&Water[{1.0 - x_methanol}]"
    rho: float = PropsSI("D", "T", T_k, "P", p_pa, fluid_str)
    cp: float = PropsSI("CPMASS", "T", T_k, "P", p_pa, fluid_str)
    lam: float = PropsSI("L", "T", T_k, "P", p_pa, fluid_str)
    return rho, cp, lam

# ---------------------------------------------------------
# B) Berechnung der Heizbandleistung pro Segment
# ---------------------------------------------------------
def segment_heating_power(ganghoehe: float, seg_length: float, d_out: float, band_power_density: float) -> float:
    """
    Berechnet die spezifische Heizleistung (W/m) eines Segments.
    ganghoehe: Wicklungsabstand [m]
    seg_length: Segmentlänge [m]
    d_out: Außendurchmesser [m]
    band_power_density: Leistungsdichte [W/m]
    """
    if ganghoehe <= 0:
        return 0.0
    turns_per_m: float = 1.0 / ganghoehe
    circumference: float = np.pi * d_out
    band_length_seg: float = turns_per_m * circumference * seg_length
    P_seg: float = band_power_density * band_length_seg  # Gesamtleistung im Segment [W]
    q_in_segment: float = P_seg / seg_length  # Heizleistung pro Meter [W/m]
    return q_in_segment

# ---------------------------------------------------------
# C) Differentialgleichungssystem für die Reaktionszone (Temperatur und Umsatz)
# ---------------------------------------------------------
def reaction_ode(_x: float, y: np.ndarray, params: ReactorParameters) -> List[float]:
    """
    Definiert das ODE-System für die Reaktionszone.
    y = [T, X] mit:
      T: Fluidtemperatur [°C]
      X: Methanol-Umsatz (0 bis 1)
    Der Heizbandeintrag fließt über die dynamisch berechnete Wandtemperatur ein:
      T_wall = max(300°C, T + q_in / (alpha_i * π * d_in))
    """
    T: float = y[0]  # Fluidtemperatur [°C]
    X: float = y[1]  # Umsatz
    m_dot: float = params.m_dot
    cp: float = params.cp
    alpha_i: float = params.alpha_i
    d_in: float = params.d_in
    q_in: float = params.q_in_segment

    # Dynamische Berechnung der Wandtemperatur
    T_wall_dynamic: float = T + q_in / (alpha_i * np.pi * d_in)
    if T_wall_dynamic < 300.0:
        T_wall_dynamic = 300.0  # Mindestwandtemperatur
    perimeter: float = np.pi * d_in  # Innenperimeter
    Q_conv: float = alpha_i * perimeter * (T_wall_dynamic - T)  # Konvektiver Wärmeaustausch

    n_dot_meoh: float = params.n_dot_meoh
    dH_reac: float = params.dH_reac
    k_reac: float = params.k_reac
    dXdx: float = k_reac * (1.0 - X)
    Q_reac: float = n_dot_meoh * dH_reac * dXdx  # Wärmeentzug durch Reaktion

    # Differentialgleichung für Fluidtemperatur (Heizbandleistung ist in T_wall_dynamic integriert)
    dTdx: float = (Q_conv - Q_reac) / (m_dot * cp)
    return [dTdx, dXdx]

# ---------------------------------------------------------
# D) Sequentielle Optimierung der Ganghöhen-Verteilung
# ---------------------------------------------------------
def sequential_optimize_ganghoehen(T_in: float, N_seg: int, L_reac: float, d_in: float, d_out: float,
                                   band_power_density: float, params_const: ReactorParameters,
                                   T_target: float = 300.0, Kp: float = 0.0002,
                                   ganghoehe_init: float = 0.015, ganghoehe_min: float = 0.0082, ganghoehe_max: float = 1.0,
                                   extra_adjust: bool = False, debug: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List]:
    """
    Optimiert den Wicklungsabstand segmentweise.
    Rückgaben:
      ganghoehen: Array der Ganghöhen [m]
      T_profile: Array der Fluidtemperaturen [°C]
      X_profile: Array der Methanol-Umsätze
      T_wall_profile: Array der dynamisch berechneten Wandtemperaturen [°C]
      debug_info: Debug-Informationen (optional)
    """
    seg_length: float = L_reac / N_seg
    ganghoehen: np.ndarray = np.zeros(N_seg)
    T_profile: np.ndarray = np.zeros(N_seg + 1)
    X_profile: np.ndarray = np.zeros(N_seg + 1)
    T_wall_profile: np.ndarray = np.zeros(N_seg + 1)

    T_profile[0] = T_in         # Startfluidtemperatur
    X_profile[0] = 0.0          # Kein Umsatz zu Beginn
    T_wall_profile[0] = 300.0   # Anfangswert: Mindestwandtemperatur
    current_ganghoehe_local: float = ganghoehe_init

    increasing_count: int = 0
    extra_factor: float = 1.0
    debug_info: List = [] if debug else None

    for i in range(N_seg):
        T_start: float = T_profile[i]
        # Heizbandeintrag im aktuellen Segment
        q_in_seg: float = segment_heating_power(current_ganghoehe_local, seg_length, d_out, band_power_density)
        # Kopie der konstanten Parameter mit aktuellem q_in_segment
        params_seg: ReactorParameters = ReactorParameters(**vars(params_const))
        params_seg.q_in_segment = q_in_seg

        # Numerische Integration für das Segment
        ode_with_params = partial(reaction_ode, params=params_seg)
        sol = solve_ivp(fun=ode_with_params,
                        t_span=(0, seg_length),
                        y0=[T_start, X_profile[i]],
                        method='RK45',
                        dense_output=True,
                        max_step=seg_length / 10.0)
        T_end: float = sol.y[0, -1]
        X_end: float = sol.y[1, -1]
        T_profile[i + 1] = T_end
        X_profile[i + 1] = X_end

        # Dynamische Berechnung der Wandtemperatur im Segment
        T_wall_seg: float = T_end + q_in_seg / (params_const.alpha_i * np.pi * d_in)
        if T_wall_seg < 300.0:
            T_wall_seg = 300.0
        T_wall_profile[i + 1] = T_wall_seg

        if extra_adjust:
            if T_end > T_start:
                increasing_count += 1
            else:
                increasing_count = 0
            extra_factor = 1.5 if increasing_count >= 3 else 1.0
        else:
            extra_factor = 1.0

        # Regelalgorithmus: Anpassung des Wicklungsabstands
        error: float = T_end - T_target
        new_ganghoehe: float = current_ganghoehe_local + Kp * error * extra_factor
        new_ganghoehe = np.clip(new_ganghoehe, ganghoehe_min, ganghoehe_max)
        ganghoehen[i] = current_ganghoehe_local

        if debug:
            debug_info.append({
                'segment': i + 1,
                'T_start': T_start,
                'T_end': T_end,
                'ganghoehe': current_ganghoehe_local
            })

        current_ganghoehe_local = new_ganghoehe

    return ganghoehen, T_profile, X_profile, T_wall_profile, debug_info

# ---------------------------------------------------------
# E) Berechnung der Wärmeverluste
# ---------------------------------------------------------
def heat_loss_total(T_wall_C: float, T_amb_C: float, length: float, d_in: float, d_wall: float, delta_iso: float,
                    lambda_iso: float = 0.073, alpha_out: float = 10.0, eps: float = 0.8) -> float:
    """
    Berechnet den Gesamtwärmeverlust (W) eines Rohrs.
    """
    T_wall_K: float = T_wall_C + CELSIUS_TO_KELVIN
    T_amb_K: float = T_amb_C + CELSIUS_TO_KELVIN
    r_outer: float = (d_in + 2 * d_wall) / 2.0
    A_outer: float = 2 * np.pi * r_outer * length
    if delta_iso <= 0.0:
        sigma: float = 5.670374419e-8
        Q_conv: float = alpha_out * A_outer * (T_wall_K - T_amb_K)
        Q_rad: float = eps * sigma * A_outer * (T_wall_K ** 4 - T_amb_K ** 4)
        Q_dot: float = Q_conv + Q_rad
    else:
        r_iso_outer: float = r_outer + delta_iso
        T_diff: float = T_wall_C - T_amb_C
        Q_dot = (2 * np.pi * lambda_iso * length * T_diff) / np.log(r_iso_outer / r_outer)
    return Q_dot

# ---------------------------------------------------------
# F) Berechnung des inneren Wärmeübergangskoeffizienten (alpha_i)
# ---------------------------------------------------------
def calc_alpha_i(T_degC: float, p_pa: float, m_dot: float, d_in: float, x_methanol: float = 0.5,
                 heating: bool = True) -> float:
    """
    Berechnet alpha_i [W/(m²·K)] mittels der Dittus-Boelter-Korrelation.
    """
    rho, cp, lam = fluid_properties(T_degC, p_pa, x_methanol)
    T_k = T_degC + CELSIUS_TO_KELVIN
    fluid_str = f"HEOS::Methanol[{x_methanol}]&Water[{1.0 - x_methanol}]"
    mu = PropsSI("V", "T", T_k, "P", p_pa, fluid_str)
    u = 0.01  # mittlere Strömungsgeschwindigkeit [m/s]
    nu = mu / rho
    Re = (u * d_in) / nu
    if Re < 2300:
        Nu = 3.66
        print(f"Laminarer Bereich erkannt (Re = {Re:.1f} < 2300). Verwende Nu = {Nu:.2f}.")
    alpha_i = Nu * lam / d_in
    return alpha_i

# ---------------------------------------------------------
# Hauptprogramm
# ---------------------------------------------------------
def main() -> None:
    # Parameterdefinitionen
    d_in: float = 0.050      # m, Innenrohrdurchmesser
    d_wall: float = 0.003    # m, Rohrwanddicke
    d_out: float = d_in + 2 * d_wall  # m, Außendurchmesser

    L_vor: float = 0.30      # m, Länge der Vorwärmzone
    L_reac: float = 0.60     # m, Länge der Reaktionszone
    N_seg: int = 3000        # Anzahl der Segmente in der Reaktionszone (hohe Auflösung)

    band_power_density: float = 200.0  # W/m, Leistungsdichte des Heizbands

    T_in: float = 200.0      # °C, Eintrittstemperatur in der Vorwärmzone
    T_amb: float = 25.0      # °C, Umgebungstemperatur
    p_reac: float = 20e5     # Pa, 20 bar
    x_meth: float = 0.5      # Massenanteil Methanol

    # Fluiddaten
    T_mean: float = 300.0    # °C, angenommene mittlere Fluidtemperatur
    rho, cp, lam_fl = fluid_properties(T_mean, p_reac, x_meth)

    # Strömungsdaten
    u: float = 0.01        # m/s, mittlere Strömungsgeschwindigkeit
    A_in: float = np.pi * (d_in ** 2) / 4.0  # Querschnittsfläche [m²]
    m_dot: float = rho * u * A_in  # Massenstrom [kg/s]

    # Wärmeübergangs- und Reaktionsdaten
    alpha_i = calc_alpha_i(T_mean, p_reac, m_dot, d_in, x_methanol=0.5, heating=False)
    alpha_i = 300  # Beispielwert, ggf. experimentell anzupassen
    n_dot_meoh: float = 0.01  # mol/s, Stoffmengenstrom Methanol
    dH_reac: float = 49200.0  # J/mol, Reaktionsenthalpie
    X_meoh: float = 0.99      # Zielumsatz von Methanol (99%)
    k_reac: float = -math.log(1 - X_meoh) / L_reac  # kinetischer Parameter [1/m]

    # Zusammenstellung der konstanten Parameter
    params_const: ReactorParameters = ReactorParameters(
        m_dot=m_dot,
        cp=cp,
        alpha_i=alpha_i,
        d_in=d_in,
        T_wall_reac=300.0,  # Initialer Platzhalter; wird dynamisch überschrieben
        n_dot_meoh=n_dot_meoh,
        dH_reac=dH_reac,
        k_reac=k_reac
    )

    # Integration der Vorwärmzone (hier wird T_wall als 300°C angenommen)
    def ode_vorwaermung(_x: float, T: np.ndarray) -> float:
        perimeter: float = np.pi * d_in
        return alpha_i * perimeter * (300.0 - T[0]) / (m_dot * cp)

    sol_vor = solve_ivp(fun=ode_vorwaermung, t_span=(0, L_vor), y0=[T_in],
                        method='RK45', dense_output=True)
    x_vor: np.ndarray = np.linspace(0, L_vor, 50)
    T_vor_profile: np.ndarray = sol_vor.sol(x_vor)[0]
    T_out_vor: float = T_vor_profile[-1]

    # Berechnung der Heizbandlänge in der Vorwärmzone
    ganghoehe_vor: float = 0.04  # [m] Beispielwert für Wicklungsabstand in der Vorwärmzone
    band_length_vor: float = L_vor * (np.pi * d_out) / ganghoehe_vor

    # Parameter für die Optimierung in der Reaktionszone
    T_target: float = 300.0  # °C, Zielfluidtemperatur
    Kp: float = 0.001        # Regelparameter (kleinere Werte dämpfen Schwankungen)
    ganghoehe_init: float = 0.01  # [m] Startwert für Wicklungsabstand
    min_abstand: float = 0.005    # [m] Mindestabstand der Heizbänder
    diameter_Heizband: float = 0.003  # [m] Durchmesser des Heizbands
    ganghoehe_min: float = min_abstand * diameter_Heizband
    ganghoehe_max: float = 1.0  # Maximale Ganghöhe wurde auf 1,0 erhöht

    # Optimierung der Ganghöhen-Verteilung in der Reaktionszone
    opt_ganghoehen, T_reac_profile, X_reac_profile, T_wall_profile, _ = sequential_optimize_ganghoehen(
        T_in=T_out_vor, N_seg=N_seg, L_reac=L_reac, d_in=d_in, d_out=d_out,
        band_power_density=band_power_density, params_const=params_const,
        T_target=T_target, Kp=Kp, ganghoehe_init=ganghoehe_init,
        ganghoehe_min=ganghoehe_min, ganghoehe_max=ganghoehe_max,
        extra_adjust=True, debug=False
    )
    x_reac: np.ndarray = np.linspace(0, L_reac, N_seg + 1)

    # Gesamtlängen-Arrays für Fluidtemperaturen und Wandtemperaturen:
    # Vorwärmzone: Wandtemperatur wird als konstant 300°C angenommen
    x_total = np.concatenate((x_vor, x_reac + L_vor))
    T_wall_vor_array = np.full_like(x_vor, 300.0)  # Vorwärmzone = 300°C
    # Reaktionszone: Verwende die dynamisch berechneten Wandtemperaturen
    T_wall_total = np.concatenate((T_wall_vor_array, T_wall_profile))

    # Berechnung der stöchiometrischen Verteilung der Reaktionsprodukte
    X: np.ndarray = X_reac_profile
    y_CH3OH: np.ndarray = (1 - X) / (2 + 2 * X)
    y_H2O: np.ndarray = (1 - X) / (2 + 2 * X)
    y_H2: np.ndarray = (3 * X) / (2 + 2 * X)
    y_CO2: np.ndarray = X / (2 + 2 * X)

    # Berechnung der Wärmeverluste für verschiedene Isolierungen
    L_total: float = L_vor + L_reac
    iso_list: List[float] = [0.0, 0.02, 0.04]
    iso_labels: List[str] = ["unisoliert", "20 mm", "40 mm"]
    Q_loss_list: List[float] = []
    for delta_i in iso_list:
        Ql: float = heat_loss_total(T_wall_C=300.0, T_amb_C=T_amb, length=L_total,
                                    d_in=d_in, d_wall=d_wall, delta_iso=delta_i,
                                    lambda_iso=0.073, alpha_out=10.0, eps=0.8)
        Q_loss_list.append(Ql)

    # Plot (a) Temperaturverlauf: Vorwärmzone + Reaktionszone (Fluidtemperaturen)
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(x_vor, T_vor_profile, 'r-', label="Vorwärmzone (Fluid)")
    ax1.plot(x_reac + L_vor, T_reac_profile, 'b-', label="Reaktionszone (Fluid)")
    ax1.set_xlabel("Reaktorlänge (gesamt) [m]")
    ax1.set_ylabel("Fluidtemperatur [°C]")
    ax1.set_title("Temperaturverlauf im Reaktor")
    ax1.grid(True)
    ax1.legend()

    # Plot (b) Umsatzverlauf in der Reaktionszone
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.plot(x_reac, X_reac_profile, 'm-', label="Umsatz (X)")
    ax2.set_xlabel("Reaktorlänge (Reaktionszone) [m]")
    ax2.set_ylabel("Umsatz X [-]")
    ax2.set_title("Methanol-Umsatz in der Reaktionszone")
    ax2.grid(True)
    ax2.legend()

    # Plot (c) Wärmeverluste (Balkendiagramm)
    fig3, ax3 = plt.subplots(figsize=(6, 4))
    barpos: np.ndarray = np.arange(len(iso_list))
    ax3.bar(barpos, Q_loss_list, width=0.4, color=["grey", "orange", "purple"])
    ax3.set_xticks(barpos)
    ax3.set_xticklabels(iso_labels)
    ax3.set_ylabel("Wärmeverlust [W]")
    ax3.set_title(f"Wärmeverluste bei T_Innenwand = 300°C, L = {L_total:.2f} m")
    ax3.grid(True, axis='y')

    # Plot (d) Optimierte Ganghöhen-Verteilung
    fig4, ax4 = plt.subplots(figsize=(8, 4))
    seg_index: np.ndarray = np.arange(1, N_seg + 1)
    ax4.bar(seg_index, opt_ganghoehen, color='b', width=0.8, label="Diskrete Ganghöhe-Werte")
    ax4.axhline(ganghoehe_min, color='r', ls='--', label="Minimal zulässige Ganghöhe")
    ax4.axhline(ganghoehe_max, color='k', ls='--', label="Maximal zulässige Ganghöhe")
    ax4.set_xlabel("Segment-Index")
    ax4.set_ylabel("Ganghöhe [m]")
    ax4.set_title("Optimierte Ganghöhen-Verteilung in der Reaktionszone")
    ax4.grid(True, axis='y')
    ax4.legend()

    # Plot (e) Stöchiometrische Verteilung der Reaktionsprodukte
    fig5, ax5 = plt.subplots(figsize=(6, 4))
    ax5.plot(x_reac, y_CH3OH, 'b-', label="CH3OH")
    ax5.plot(x_reac, y_H2O, 'c-', label="H2O")
    ax5.plot(x_reac, y_H2, 'r-', label="H2")
    ax5.plot(x_reac, y_CO2, 'g-', label="CO2")
    ax5.set_xlabel("Reaktorlänge (Reaktionszone) [m]")
    ax5.set_ylabel("Molare Fraktion")
    ax5.set_title("Stöchiometrische Verteilung der Reaktionsprodukte")
    ax5.grid(True)
    ax5.legend()

    # Neuer Plot (f): Verlauf der Wandtemperatur über die gesamte Reaktorlänge
    # Vorwärmzone: Konstante 300°C, Reaktionszone: Dynamisch berechnete Wandtemperaturen
    fig6, ax6 = plt.subplots(figsize=(6, 4))
    ax6.plot(x_total, T_wall_total, 'g-', label="Wandtemperatur")
    # Zeichne die Mindestwandtemperaturlinie (300°C) als rot gestrichelte Linie
    ax6.axhline(300, color='r', linestyle='--', label="Mindestwandtemperatur (300°C)")
    # Vertikale Linie zur Kennzeichnung des Übergangs von Vorwärm- zu Reaktionszone
    ax6.axvline(L_vor, color='k', linestyle='--', label="Übergang Vorwärm-/Reaktionszone")
    ax6.set_xlabel("Reaktorlänge (gesamt) [m]")
    ax6.set_ylabel("Wandtemperatur [°C]")
    ax6.set_title("Verlauf der Wandtemperatur im Reaktor")
    ax6.grid(True)
    ax6.legend()

    plt.tight_layout()
    plt.show()

    # Konsolenausgabe der Ergebnisse
    print("\n=================== Ergebnisse ===================")
    print(f"Fluid: ~{T_mean} °C, p = {p_reac / 1e5:.1f} bar, xMeOH = {x_meth}")
    print(f"rho = {rho:.3f} kg/m³, cp = {cp:.2f} J/(kgK), lam = {lam_fl:.5f} W/(mK)")
    print(f"m_dot = {m_dot:.6f} kg/s (u = {u} m/s)\n")
    print("--- Vorwärmzone ---")
    print(f"L = {L_vor} m, T_in = 200°C  => T_out = {T_out_vor:.2f}°C")
    print(f"Heizband Leistung (Vorwärmzone): {band_power_density:.1f} W/m")
    print(f"Berechnete Heizbandlänge in der Vorwärmzone (bei Ganghöhe = {ganghoehe_vor:.4f} m): {band_length_vor:.2f} m")
    print("\n--- Reaktionszone ---")
    print(f"L = {L_reac}")
    max_temp = np.max(T_reac_profile)
    min_temp = np.min(T_reac_profile)
    print(f"Maximale Fluidtemperatur in der Reaktionszone: {max_temp:.2f} °C")
    print(f"Minimale Fluidtemperatur in der Reaktionszone: {min_temp:.2f} °C")
    print(f"Heizband Leistung (Reaktionszone): {band_power_density:.1f} W/m")
    seg_len: float = L_reac / N_seg
    total_bandlen_reac: float = 0.0
    for ganghoehe in opt_ganghoehen:
        turns_m: float = 1.0 / ganghoehe
        circumference: float = np.pi * d_out
        total_bandlen_reac += turns_m * circumference * seg_len
    total_power_reac: float = band_power_density * total_bandlen_reac
    print(f"Heizbandlänge in der Reaktionszone: {total_bandlen_reac:.2f} m")
    print(f"-> Damit Heizleistung ~ {total_power_reac:.1f} W (Reaktionszone)")

if __name__ == "__main__":
    main()