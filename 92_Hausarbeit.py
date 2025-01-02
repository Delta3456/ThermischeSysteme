"""
Donnerstag, 02.01.25
@author Janik Focken
--------------------------------
Zweite Hausarbeit - Instationäre Wärmeleitung
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp




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
    Simulierung der Temeratur der Umgebung nach einer Sinusfunktion.
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

def radiation(T_surf_C, T_amb_C, A, eps=0.9, sigma=5.67e-8):
    """
    Berechnet den Strahlungswärmestrom Q_rad [W] von einer Oberfläche
    T_surf_C, T_amb_C: Oberflächentemperatur und Umgebungstemperatur in °C
    A: Fläche
    eps: Emissionsgrad
    sigma: Stefan-Boltzmann-Konstante
    """
    T_surf_K = T_surf_C + 273.15
    T_amb_K  = T_amb_C  + 273.15
    return eps * sigma * A * (T_surf_K**4 - T_amb_K**4)