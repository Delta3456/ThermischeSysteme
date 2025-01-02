"""
Donnerstag, 02.01.25
@author Janik Focken
--------------------------------
Zweite Hausarbeit - Instationäre Wärmeleitung
"""
import numpy as np
import matplotlib.pyplot as plt
import copy
from scipy.integrate import solve_ivp

# Parameter -------------------------------------------------------------------
# Stoffeigenschaften Aluminium → Noch Quelle?/Prüfen/Temp/druck
rho_alu = 2700.0
cp_alu = 900.0
kappa_alu = 200.0

# Stoffeigenschaften Luft
rho_luft = 1.2
cp_luft = 1000.0

# Stoffeigenschaften Kunststoff
rho_kunst = 1200.0
cp_kunst = 1500.0
kappa_kunst = 0.2

# Relais, vereinfacht als Aluminium
H_rel = 0.023
B_rel = 0.0457
L_rel = 0.0584
V_rel = L_rel * B_rel * H_rel
m_rel = rho_alu * V_rel
# Oberfläche Relais, ohne Boden zur Platte, da Kontakt zur Platte
A_rel = 2 * ( B_rel * H_rel + L_rel * H_rel) + (L_rel * B_rel)

# Montageplatte, Aluminium
L_platte = 0.1
B_platte = 0.1
d_platte = 0.01
V_platte = L_platte * B_platte * d_platte
m_platte = rho_alu * V_platte
A_platte_out = L_platte * B_platte

# Luft im Gehäuse
V_luft = L_platte * B_platte * (H_rel + 0.025) - V_rel
m_luft = rho_luft * V_luft

# Gehäuse, Kunststoff
dicke_geh = 0.003
# Aussenfläche vom Gehäuse ohne Boden, Vereinfacht ist A_geh_out = A_geh_in
A_geh_out = 2 * (B_platte * (H_rel + 0.025)) + 2 * (L_platte * (H_rel + 0.025)) + (L_platte * B_platte)
V_geh = A_geh_out * dicke_geh
m_geh = rho_kunst * V_geh

# Wärmeübergangkoeffizenten
alpha_rel_luft = 10.0
alpha_luft_geh = 10.0
alpha_geh_out = 5.0
alpha_platte_out = 5.0

# Regelung Ventilator
use_regelung = False
track_fan_power = False
fan_power = 0.0 # Aktuelle Ventilatorleistung
fan_alpha = 0.0 # Akteller Wert
fan_power_fest = 0.0
fan_alpha_fest = 5.0
fan_power1 = 100.0
fan_alpha1 = 30.0
fan_power2 = 250.0
fan_alpha2 = 60.0
# Temperaturen, wann sich der Ventilator einschaltet
fan_stage = 0
T_stage1_on = 60.0
T_stage1_off = 55.0
T_stage2_on = 75.0
T_stage2_off = 70.0


# Weitere
Q_rel = 30.0  # 30 W Verlustleistung
R_th_RP = 0.5  # Kontaktwiderstand Relais-Platte ->?????? Wofür?, oder perfekt annehmen
t_end = 72*3600 # Simululationszeit
steps = 2000 # Berechnungsschritte


# Parameter-Dictonary
params_dict = {
    "m_rel": m_rel, "cp_alu": cp_alu,
    "m_platte": m_platte, "cp_P": cp_alu,
    "m_luft": m_luft, "cp_luft": cp_luft,
    "m_geh": m_geh, "cp_kunst": cp_kunst,

    "alpha_rel_luft": alpha_rel_luft,
    "alpha_luft_geh": alpha_luft_geh,
    "alpha_geh_out": alpha_geh_out,
    "alpha_platte_out": alpha_platte_out,

    "Q_rel": Q_rel,
    "R_th_RP": R_th_RP,

    "A_rel": A_rel,
    "A_platte_out": A_platte_out,
    "A_geh_out": A_geh_out,

    "use_regelung": use_regelung,
    "track_fan_power": track_fan_power,
    "fan_power": fan_power,
    "fan_alpha": fan_alpha,
    "fan_power_fest": fan_power_fest,
    "fan_alpha_fest": fan_alpha_fest,
    "fan_power1": fan_power1,
    "fan_alpha1": fan_alpha1,
    "fan_power2": fan_power2,
    "fan_alpha2": fan_alpha2,
    "fan_stage": fan_stage,
    "T_stage1_on": T_stage1_on,
    "T_stage1_off": T_stage1_off,
    "T_stage2_on": T_stage2_on,
    "T_stage2_off": T_stage2_off,

}


# Funktionen -------------------------------------------------------------------------
def biot_number(alpha, L_char, kappa):
    """
    Berechnet die Biot-Zahl: Bi = alpha * L_char / kappa.
    alpha: Wärmeübergangskoeffizienten
	L_char: charakteristische Länge
	kappa: Wärmeleitfähigkeit
    """
    return alpha * L_char / kappa

def check_biot(name, alpha, L_char, kappa):
    """
    Prüft die Biot-Zahl, denn für die Methode der Blockkapazitäten (lumped capacity analysis)
    sollte Bi << 1 sein, daher die Bedingung Bi =< 0.1
    """
    bi = biot_number(alpha, L_char, kappa)
    if bi >= 0.1:
        print(f"[WARNUNG] {name}: Biot-Zahl = {bi:.3f} (>= 0.1) -> "
              f"Methode der Blockkapazitäten evtl. ungenau.")
    else:
        print(f"[OK] {name}: Biot-Zahl = {bi:.3f} < 0.1.")

def T_umgebung_sinus(t):
    """
    Simulieren der Temperature der Umgebung nach einer Sinusfunktion.
    Zeit wird in Sekunden übergeben
    y=A*sin(B*t+C)+D
    """
    T_umgebung_max = 9
    T_umgebung_min = -4
    A = (T_umgebung_max - T_umgebung_min) / 2
    B = 2*np.pi/24 # Eine Welle ist 24h lang
    # C=0, keine Phasenverschiebung
    D = (T_umgebung_max + T_umgebung_min) / 2
    hours = t / 3600.0
    return A * np.sin(hours * B) + D

def energiebilanzen(t, y, params):
    """
    Energiebilanzen und Wärmeströme aufstellen
    t: Zeit
    y: Temperaturen
    params: Parameter
    """
    T_rel, T_platte, T_luft, T_geh = y
    T_u = T_umgebung_sinus(t)

    if params["use_regelung"]:
        fan_stage = params["fan_stage"]
        T_stage1_on   = params["T_stage1_on"]
        T_stage1_off  = params["T_stage1_off"]
        T_stage2_on   = params["T_stage2_on"]
        T_stage2_off  = params["T_stage2_off"]

        # Regelungslogik
        if fan_stage == 0:
            if T_rel > T_stage1_on:
                fan_stage = 1
        elif fan_stage == 1:
            if T_rel > T_stage2_on:
                fan_stage = 2
            elif T_rel < T_stage1_off:
                fan_stage = 0
        elif fan_stage == 2:
            if T_rel < T_stage2_off:
                fan_stage = 1

        params["fan_stage"] = fan_stage

        if fan_stage == 0:
            alpha_platte_out = params["alpha_platte_out"]
            params["fan_power"] = 0.0
        elif fan_stage == 1:
            alpha_platte_out = params["fan_alpha1"]
            params["fan_power"] = params["fan_power1"]
        else: # fan_stage == 2
            alpha_platte_out = params["fan_alpha2"]
            params["fan_power"] = params["fan_power2"]

    else:
        # Keine Regelung
        alpha_platte_out = params["fan_alpha_fest"]
        params["fan_power"] = params["fan_power_fest"]

    # Debbuging
    """
    if t % (t_end // 10) < 1:  
        print(
            f"t={t / 3600:.1f}h, use_regelung={params['use_regelung']}, fan_alpha_fest={params['fan_alpha_fest']}, fan_power_fest={params['fan_power_fest']}, fan_power={params['fan_power']}")
    """

    # Wärmeströme -----------------------------------------------
    # Relais, gibt nur Wärme ab an die Luft und an die Platte
    Q_rel_platte = (T_rel - T_platte) / params["R_th_RP"]
    Q_rel_luft = params["alpha_rel_luft"] * params["A_rel"] * (T_rel - T_luft)
    dT_rel_dt = ( params["Q_rel"] - Q_rel_luft - Q_rel_platte ) / ( params["m_rel"] * params["cp_alu"] )

    # Platte, Wärmezufuhr von dem Relais und Abfuhr in die Umgebung
    Q_platte_u = alpha_platte_out * params["A_platte_out"] * (T_platte - T_u)
    dT_platte_dt = ( Q_rel_platte - Q_platte_u ) / ( params["m_platte"]*params["cp_alu"] )

    # Luft, Wärmezufuhr von dem Relais und Abfuhr zum Gehäuse
    Q_luft_geh = params["alpha_luft_geh"] * params["A_geh_out"] * (T_luft - T_geh)
    dT_luft_dt = ( Q_rel_luft - Q_luft_geh ) / ( params["m_luft"]*params["cp_luft"] )

    # Gehäuse, Wärmezufuhr von der Luft und Abfuhr in die Umgebung
    Q_geh_u = params["alpha_geh_out"] * params["A_geh_out"] * (T_geh - T_u)
    dT_geh_dt = ( Q_luft_geh - Q_geh_u ) / ( params["m_geh"]*params["cp_kunst"] )

    return [dT_rel_dt, dT_platte_dt, dT_luft_dt, dT_geh_dt]

def berechnung(params, t_end=t_end, steps=steps):
    """
    Löst, die Wärmebilanzen und gibt zeiten und Temeraturarrays zurück
    """
    # Anfangswerte: 5°C
    y0 = [5.0, 5.0, 5.0, 5.0]
    t_eval = np.linspace(0, t_end, steps)

    if params.get("track_fan_power"):
        params["fan_power_array"] = []

    def store_fan_power(t,y):
        if params.get("track_fan_power"):
            # Aktuelle Ventilatorleistung ablesen und in fan_power_array eintragen
            params["fan_power_array"].append((t, params["fan_power"]))

    def berechnung_mit_tracking(t,y):
        store_fan_power(t,y)
        return energiebilanzen(t, y, params)

    # Solve
    sol = solve_ivp(
        fun=berechnung_mit_tracking,
        t_span=(0, t_end),
        y0=y0,
        t_eval=t_eval,
        method="RK45",
        max_step=10.0
    )

    # Letzten Punkt nochmals speichern
    store_fan_power(sol.t[-1], sol.y[:, -1])

    return sol.t, sol.y[0], sol.y[1], sol.y[2], sol.y[3]


# Hauptberechnung -------------------------------------------------------------------------
# Biot-Zahl-Prüfung
# Die Characterristische Länge für die Biot-Zahl ist, das Volumen durch die Fläche
Lc_rel = V_rel / A_rel
Lc_geh = V_geh / A_geh_out
check_biot("Relais", alpha_rel_luft, Lc_rel, kappa_alu)
check_biot("Platte", alpha_platte_out, d_platte, kappa_alu)
check_biot("Gehäusewand", alpha_geh_out, Lc_geh, kappa_kunst)

# Szenarien -----------
# (1) Ohne Ventilator
scenario_no_fan = copy.deepcopy(params_dict)
scenario_no_fan["use_regelung"] = False
scenario_no_fan["fan_alpha_fest"] = alpha_platte_out
scenario_no_fan["fan_power_fest"] = 0.0

# (2) Ventilator Stufe 1 (dauerhaft)
scenario_fan1 = copy.deepcopy(params_dict)
scenario_fan1["use_regelung"] = False
scenario_fan1["fan_alpha_fest"] = fan_alpha1
scenario_fan1["fan_power_fest"] = fan_power1

# (2) Ventilator Stufe 2 (dauerhaft)
scenario_fan2 = copy.deepcopy(params_dict)
scenario_fan2["use_regelung"] = False
scenario_fan2["fan_alpha_fest"] = fan_alpha2
scenario_fan2["fan_power_fest"] = fan_power2

# (4) Regelung
scenario_regel = copy.deepcopy(params_dict)
scenario_regel["use_regelung"] = True
scenario_regel["fan_stage"] = 0
scenario_regel["track_fan_power"] = True

# Berechnung
print("\n=== (1) Ohne Ventilator ===")
t_nf, T_rel_nf, T_platte_nf, T_luft_nf, T_geh_nf = berechnung(scenario_no_fan, t_end, steps)

print("\n=== (2) Ventilator Stufe 1 ===")
t_f1, T_rel_f1, T_platte_f1, T_luft_f1, T_geh_f1 = berechnung(scenario_fan1, t_end, steps)

print("\n=== (3) Ventilator Stufe 2 ===")
t_f2, T_rel_f2, T_platte_f2, T_luft_f2, T_geh_f2 = berechnung(scenario_fan2, t_end, steps)

print("\n=== (4) Mehrstufige Regelung ===")
t_regel, T_rel_regel, T_platte_regel, T_luft_regel, T_geh_regel = berechnung(scenario_regel, t_end, steps)

# Ventilator Energie --------
#  Ohne Ventilator
E_fan_nf = 0.0  # 0 W

# Ventilator Stufe 1 (dauerhaft)
E_fan_f1_Wh = 100.0 * 72  # 100 W * 72h
E_fan_f1_kWh = E_fan_f1_Wh / 1000.0

# Ventilator Stufe 2 (dauerhaft)
E_fan_f2_Wh = 250.0 * 72  # 250 W * 72h
E_fan_f2_kWh = E_fan_f2_Wh / 1000.0

# (4) Regelung
fan_array = scenario_regel["fan_power_array"]  # Liste von (t, fan_power)
E_fan_sum_J = 0.0

# Energie summieren
for i in range(len(fan_array) - 1):
    t_i, p_i = fan_array[i]
    t_ip1, _ = fan_array[i + 1]
    dt_i = t_ip1 - t_i  # Zeitdifferenz
    E_fan_sum_J += p_i * dt_i  # p_i [W], dt_i [s]

E_fan_regel_Wh = E_fan_sum_J / 3600.0
E_fan_regel_kWh = E_fan_regel_Wh / 1000.0

print("\nVentilator-Energie:")
print(f"  (1) Ohne Ventilator:        {E_fan_nf:.3f} kWh")
print(f"  (2) Stufe 1 (dauer):        {E_fan_f1_kWh:.3f} kWh")
print(f"  (3) Stufe 2 (dauer):        {E_fan_f2_kWh:.3f} kWh")
print(f"  (4) Mehrstufige Regelung:   {E_fan_regel_kWh:.3f} kWh")

# Plotten ----------
dt = t_nf[1] - t_nf[0]  # Zeitschrittabstand (konstant)
#  Plot 1: Relais-Temperatur
plt.figure(figsize=(10,6))
plt.plot(t_nf/3600,  T_rel_nf,  label="(1) T_rel noFan")
plt.plot(t_f1/3600,  T_rel_f1,  label="(2) T_rel Fan1")
plt.plot(t_f2/3600,  T_rel_f2,  label="(3) T_rel Fan2")
plt.plot(t_regel/3600, T_rel_regel, label="(4) T_rel regel")

T_u_plot = [T_umgebung_sinus(ti) for ti in t_nf]
plt.plot(t_nf/3600, T_u_plot, 'k--', label="T_Umgebung")
plt.axhline(80, color='r', ls=':', label="80°C-Grenze")

plt.title("Temperatur des Relais in verschiedenen Szenarien")
plt.xlabel("Zeit [h]")
plt.ylabel("T [°C]")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Plot 2: Montageplatte
plt.figure(figsize=(10,6))
plt.plot(t_nf/3600,  T_platte_nf,  label="(1) T_platte noFan")
plt.plot(t_f1/3600,  T_platte_f1,  label="(2) T_platte Fan1")
plt.plot(t_f2/3600,  T_platte_f2,  label="(3) T_platte Fan2")
plt.plot(t_regel/3600, T_platte_regel, label="(4) T_platte regel")
plt.plot(t_nf/3600, T_u_plot, 'k--', label="T_Umgebung")
plt.title("Temperatur der Montageplatte in verschiedenen Szenarien")
plt.xlabel("Zeit [h]")
plt.ylabel("T [°C]")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Plot 3: Luft im Gehäuse
plt.figure(figsize=(10,6))
plt.plot(t_nf/3600,  T_luft_nf,  label="(1) T_luft noFan")
plt.plot(t_f1/3600,  T_luft_f1,  label="(2) T_luft Fan1")
plt.plot(t_f2/3600,  T_luft_f2,  label="(3) T_luft Fan2")
plt.plot(t_regel/3600, T_luft_regel, label="(4) T_luft regel")
plt.plot(t_nf/3600, T_u_plot, 'k--', label="T_Umgebung")
plt.title("Temperatur der Luft im Gehäuse in verschiedenen Szenarien")
plt.xlabel("Zeit [h]")
plt.ylabel("T [°C]")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Plot 4: Kunststoffgehäuse (T_G) ---
plt.figure(figsize=(10,6))
plt.plot(t_nf/3600,  T_geh_nf,  label="(1) T_geh noFan")
plt.plot(t_f1/3600,  T_geh_f1,  label="(2) T_geh Fan1")
plt.plot(t_f2/3600,  T_geh_f2,  label="(3) T_geh Fan2")
plt.plot(t_regel/3600, T_geh_regel, label="(4) T_geh regel")
plt.plot(t_nf/3600, T_u_plot, 'k--', label="T_Umgebung")
plt.title("Temperatur des Kunststoffgehäuses in verschiedenen Szenarien")
plt.xlabel("Zeit [h]")
plt.ylabel("T [°C]")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.show()




