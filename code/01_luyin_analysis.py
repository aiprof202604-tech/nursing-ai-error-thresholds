"""
Lu & Yin 2021 Experiment 2 reanalysis.

Input:  expTwoFinalPredictionsValid1125.csv (from https://github.com/ZhuoranLu/Trustworthy-ML)
Output: worker-level reliance means at A = 0.50 and A = 0.80
Used in: Phase 2 validation and Phase 3 model calibration
"""
import pandas as pd
import numpy as np
from scipy import stats

df = pd.read_csv('expTwoFinalPredictionsValid1125.csv', index_col=0)

# Normalize boolean columns
for col in ['selfCorrect','finalCorrect','finalAgreement','mlCorrect','switch']:
    df[col] = df[col].astype(str).str.upper().map({'TRUE':1,'FALSE':0})

print(f"Total trials: {len(df)}")
print(f"Unique workers: {df['workerId'].nunique()}")
print(f"acc levels: {sorted(df['acc'].unique())}")

# Worker-level means
worker = df.groupby(['workerId','acc']).agg(
    finalAgreement=('finalAgreement','mean'),
    switch_overall=('switch','mean'),
).reset_index()

# Conditional switch: trials where human initially disagreed with AI
cond = df[df['agreement']==0].groupby(['workerId','acc'])['switch'].mean().reset_index()
cond.columns = ['workerId','acc','switch_given_disagree']
worker = worker.merge(cond, on=['workerId','acc'], how='left')

print("\nBy acc level (worker-level M ± SD, N workers):")
for a in [50, 80]:
    sub = worker[worker['acc']==a]
    print(f"  acc={a}% (N={len(sub)}):")
    for m in ['finalAgreement','switch_overall','switch_given_disagree']:
        print(f"    {m:24s}: {sub[m].mean():.3f} ± {sub[m].std():.3f}")

print("\nStatistical tests (acc=50 vs acc=80):")
for m in ['finalAgreement','switch_overall','switch_given_disagree']:
    g50 = worker[worker['acc']==50][m].dropna()
    g80 = worker[worker['acc']==80][m].dropna()
    t, p = stats.ttest_ind(g50, g80)
    print(f"  {m:24s}: Δ(80-50)={g80.mean()-g50.mean():+.3f}, t={t:.2f}, p={p:.4f}")
