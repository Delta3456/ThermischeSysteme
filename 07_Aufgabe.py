"""
Samstag, 23.11.24
@author Janik Focken
--------------------------------
Aufgabe 7 - Energiespeicher
Instationäre gekoppelte homogene Systeme


Formeln:

Annahme: Im Zeitintervall der DGL gilt:
T_sp = konstant

C_min/c_max = 0,
epsilon = 1 - np.exp(-NTU)
epsilon = (T_l_out - T_l_in)/(T_l_out - T_sp)
T_l_out = T_sp - (T_l_out - T_l_in) * np.exp(-NTU)
W_dot = m_dot_l * c_l
Q_dot = W_dot * (T_l_out - T_w_in)
m_sp * c_w * (dT_sp/dt) = Q_dot
dT_sp/dt = (m_dot_l * c_l * dT_max)/(m_sp * c_w)Annahme: Im Zeitintervall der DGL gilt:
T_sp = konstant

C_min/c_max = 0,
epsilon = 1 - np.exp(-NTU)
epsilon = (T_l_out - T_l_in)/(T_l_out - T_sp)
T_l_out = T_sp - (T_l_out - T_l_in) * np.exp(-NTU)
W_dot = m_dot_l * c_l
Q_dot = W_dot * (T_l_out - T_w_in)
m_sp * c_w * (dT_sp/dt) = Q_dot
dT_sp/dt = (m_dot_l * c_l * dT_max)/(m_sp * c_w)
"""
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Gegebene Werte
L_sp = 0.5  # m, Kantenlänge des Würfels
D_r = 0.04  # m, Durchmesser des Rohres
L_r = 16.0  # m, Länge des Rohres
T_u = 20.0  # °C, Umgebungstemperatur
T_l_in = 90.0  # °C, Einströmtemperatur der Luft
m_dot_l = 0.025  # kg/s, Massenstrom der Luft
alpha_i = 100.0  # W/(m^2 K), Wärmeübergangskoeffizient innen
alpha_a = 3.0  # W/(m^2 K), Wärmeübergangskoeffizient außen
rho_w = 1000.0  # kg/m^3, Dichte des Wassers
c_w = 4180.0  # J/(kg K), spezifische Wärmekapazität des Wassers
c_l = 1001.0  # J/(kg K), spezifische Wärmekapazität der Luft

# Berechnete Werte
A_r = np.pi * D_r * L_r  # Rohrfläche
A_sp = 6 * L_sp**2  # Oberfläche des Würfels
V_sp = L_sp**3  # Volumen des Würfels
m_sp = rho_w * V_sp  # Masse des Wassers im Speicher
