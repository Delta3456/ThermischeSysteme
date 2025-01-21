# -*- coding: utf-8 -*-
"""
Beispielskript zur Erzeugung zufälliger Daten für eine Carnot-Wärmekraftmaschine
mit irreversibler Wärmeübertragung auf der Hochtemperaturseite.
Es werden Pareto-Optimallösungen bzgl. Wirkungsgrad (max), Leistung (max)
und Wärmetauscherfläche (min) gesucht.

Autor: CHAT GPT o1
Datum: 2025-01-18
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# 1) Parameter festlegen
# -----------------------------
T_source = 1000.0   # [K] Temperatur Wärmequelle
T_sink   = 300.0    # [K] Temperatur Wärmesenke
U        = 10.0     # [W/(m²·K)] Wärmeübertragungskoeffizient
N_points = 1000     # Anzahl zufällig zu generierender Punkte

# -----------------------------
# 2) Zufallsdaten erzeugen
# -----------------------------
# DeltaT kann z.B. zwischen 10 K und 300 K variieren
# A soll zwischen 1 m² und 100 m² schwanken
np.random.seed(42)  # Reproduzierbare Zufallszahlen
DeltaT_values = np.random.uniform(low=10.0, high=300.0, size=N_points)
A_values      = np.random.uniform(low=1.0 , high=100.0, size=N_points)

# -----------------------------
# 3) Wirkungsgrad, Leistung berechnen
# -----------------------------
eta_list = []
P_el_list = []

for dT, A in zip(DeltaT_values, A_values):
    # a) Temperatur im Carnot-Prozess
    T_engine_hot = T_source - dT  # [K]

    # b) Wirkungsgrad
    # (Achtung: wenn dT zu groß wird, kann T_engine_hot < T_sink werden;
    #  für dieses Beispiel ignorieren wir das oder filtern es. Real müsste man
    #  unzulässige Punkte ausschließen.)
    if T_engine_hot <= T_sink:
        eta = 0.0
    else:
        eta = 1.0 - (T_sink / T_engine_hot)

    # c) Wärmestrom
    Q_dot = U * A * dT  # W

    # d) Leistung
    P_el = Q_dot * eta  # W

    eta_list.append(eta)
    P_el_list.append(P_el)

# In ein NumPy-Array packen, Spalten: [eta, P_el, A]
data_array = np.column_stack((eta_list, P_el_list, A_values))

# -----------------------------
# 4) Pareto-Optimalität
# -----------------------------
# Wir importieren die gegebene Pareto-Funktion oder definieren sie hier erneut
def is_pareto_efficient(costs, sense):
    """
    Find the pareto-efficient points
    Parameters
    ----------
    costs : numpy array
        An (n_points, n_costs) array
    sense : list
        List of "min" or "max" for each objective
    Returns
    -------
    numpy array
        A boolean array of pareto-efficient points
    """
    costs_modified = costs.copy()
    for i, direction in enumerate(sense):
        if direction == "max":
            costs_modified[:, i] = -costs_modified[:, i]

    is_efficient = np.ones(costs_modified.shape[0], dtype=bool)
    for i in range(costs_modified.shape[0]):
        for j in range(costs_modified.shape[0]):
            if i != j and np.all(costs_modified[j] <= costs_modified[i]):
                is_efficient[i] = False
                break
    return is_efficient

# Ziele: [eta (max), P_el (max), A (min)]
sense_list = ["max", "max", "min"]

# Pareto-Maske ermitteln
pareto_mask = is_pareto_efficient(data_array, sense_list)

# -----------------------------
# 5) DataFrame erstellen
# -----------------------------
df = pd.DataFrame(data_array, columns=["eta", "P_el", "A"])
df["DeltaT"] = DeltaT_values
df["Pareto"] = pareto_mask

# -----------------------------
# 6) Ergebnisse visualisieren
# -----------------------------
# Beispiel: 2D-Plot Leistung vs. Wirkungsgrad, Farbcode Fläche
sns.set_style("whitegrid")
plt.figure(figsize=(8, 6))
scatter = plt.scatter(
    df["eta"], df["P_el"],
    c=df["A"], cmap="viridis",
    alpha=0.7, edgecolor="k", marker="o"
)
plt.colorbar(scatter, label="Wärmeübertragerfläche [m²]")
plt.xlabel("Wirkungsgrad η [-]")
plt.ylabel("Leistung P_el [W]")
plt.title("Zufällige Stichprobe & Pareto-Optimallösungen (fett hervorgehoben)")

# Pareto-Punkte extra markieren
pareto_points = df[df["Pareto"]]
plt.scatter(pareto_points["eta"], pareto_points["P_el"],
            c="red", marker="s", edgecolor="black",
            label="Pareto-optimal")

plt.legend()
plt.tight_layout()
plt.show()

# -----------------------------
# 7) Speichern der Pareto-Punkte
# -----------------------------
pareto_points.to_csv("pareto_points.csv", index=False)
print("Pareto-optimale Punkte wurden in 'pareto_points.csv' gespeichert.")
print("Anzahl Pareto-optimaler Lösungen:", pareto_points.shape[0])

# -----------------------------
# Zusammenfassung / Empfehlung
# -----------------------------
"""
Aus thermodynamischer Sicht erhalten wir eine Menge an Kompromisslösungen:
- Sehr hohe Wirkungsgrade (η->max) erfordern kleine DeltaT und damit kleine Wärmestromraten,
  was die absolute Leistung P_el senkt.
- Hohe Leistung P_el bekommt man bei größeren DeltaT und/oder größerer Fläche,
  was jedoch den Wirkungsgrad reduzieren kann.
- Kleine Fläche A minimiert zwar Kosten/Gewicht, schränkt jedoch P_el ein.

Die Pareto-Lösungen bilden eine "Front" an möglichen Kompromissen. 
Welche Lösung in der Praxis gewählt wird, hängt z.B. von Investitionskosten, 
geplanter Betriebsstrategie oder technischen Einschränkungen ab.
"""