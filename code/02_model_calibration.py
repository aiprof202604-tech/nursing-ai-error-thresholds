"""
Phase 3: Linear reliance model calibration against 9 empirical data points
from three independent AI-assisted decision-making studies.

Calibrated parameters: β₀, β_A
Fixed from literature: β_E=-0.20, β_C=0.20, β_AE=0.10, β_AC=0.10

Output: β₀ ≈ 0.471, β_A ≈ 0.201, RMSE ≈ 0.054
"""
import numpy as np
from scipy.optimize import minimize

BE, BC, BAE, BAC = -0.20, 0.20, 0.10, 0.10

def reliance_pred(A, E, C, b0, bA):
    rel = b0 + bA*A + BE*E + BC*C + BAE*(A*E) + BAC*(A*C)
    return np.clip(rel, 0, 1)

# 9 empirical data points from 3 studies
# All E=0 (laypeople crowdworkers), C=0.5 (moderate complexity)
targets = [
    # Lu & Yin 2021 Exp 2 (raw GitHub data)
    {'A':0.50,'E':0.0,'C':0.5,'obs':0.642,'n':252,'study':'Lu&Yin_Exp2'},
    {'A':0.80,'E':0.0,'C':0.5,'obs':0.664,'n':214,'study':'Lu&Yin_Exp2'},
    # Yin et al. 2019 Exp 3 Phase 2 (Fig 5a visual read)
    {'A':0.55,'E':0.0,'C':0.5,'obs':0.795,'n':170,'study':'Yin19_Exp3'},
    {'A':0.80,'E':0.0,'C':0.5,'obs':0.830,'n':170,'study':'Yin19_Exp3'},
    {'A':1.00,'E':0.0,'C':0.5,'obs':0.833,'n':170,'study':'Yin19_Exp3'},
    # Yin et al. 2019 Exp 1 Phase 1 (Fig 3a visual read)
    {'A':0.60,'E':0.0,'C':0.5,'obs':0.745,'n':200,'study':'Yin19_Exp1_P1'},
    {'A':0.70,'E':0.0,'C':0.5,'obs':0.760,'n':200,'study':'Yin19_Exp1_P1'},
    {'A':0.90,'E':0.0,'C':0.5,'obs':0.800,'n':200,'study':'Yin19_Exp1_P1'},
    {'A':0.95,'E':0.0,'C':0.5,'obs':0.820,'n':200,'study':'Yin19_Exp1_P1'},
]

def loss(params):
    b0, bA = params
    ss = 0.0
    for t in targets:
        pred = reliance_pred(t['A'], t['E'], t['C'], b0, bA)
        w = np.sqrt(t['n'])
        ss += (w*(pred - t['obs']))**2
    return ss

res = minimize(loss, x0=[0.5, 0.3], method='Nelder-Mead',
               options={'xatol':1e-6,'fatol':1e-8})
b0, bA = res.x
print(f"β₀ = {b0:.4f}")
print(f"β_A = {bA:.4f}")

# Fit quality
print(f"\n{'Study':18s} {'A':>5s} {'Obs':>7s} {'Pred':>7s} {'Resid':>8s}")
preds = []
for t in targets:
    p = reliance_pred(t['A'], t['E'], t['C'], b0, bA)
    preds.append(p)
    print(f"{t['study']:18s} {t['A']:>5.2f} {t['obs']:>7.3f} {p:>7.3f} {p-t['obs']:>+8.3f}")

obs = np.array([t['obs'] for t in targets])
pred = np.array(preds)
rmse = np.sqrt(np.mean((obs-pred)**2))
r2 = 1 - np.sum((obs-pred)**2)/np.sum((obs-obs.mean())**2)
print(f"\nRMSE = {rmse:.4f}")
print(f"R² (unweighted) = {r2:.3f}")
