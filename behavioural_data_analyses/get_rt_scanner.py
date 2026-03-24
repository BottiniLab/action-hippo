import os
import glob

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import ttest_rel
from statsmodels.stats.anova import AnovaRM

# =============================================================================
# Paths
# =============================================================================

path = '/data/pt_02747_hippo/2_day_two/4_scanner/responses/probe_trials/'
# path = '/data/pt_02747_hippo/1_day_one/2_eyetracker/responses/probe_trials/'
path_subs = '/data/pt_02747/action_hippo/data/derivatives/'

# =============================================================================
# Get valid subject list
# =============================================================================

subs = os.listdir(path_subs)
subs = [sub for sub in subs if 'sub' in sub]
subs = [sub for sub in subs if os.path.isdir(os.path.join(path_subs, sub))]
subs.sort()

# if second-last digit is '0', remove it in the name
good_subs_list = [sub[:-2] + sub[-1] if sub[-2] == '0' else sub for sub in subs]
good_subs_list = [sub + '_' for sub in good_subs_list]

# =============================================================================
# Get relevant CSV files
# =============================================================================

all_files = glob.glob(os.path.join(path, "*.csv"))
all_files = [f for f in all_files if any(sub in f for sub in good_subs_list)]
all_files.sort()

# =============================================================================
# Read and process files
# =============================================================================

dfs = []
num_missed = []

for file in all_files:
    df = pd.read_csv(file)

    # make participant a string
    df['participant'] = df['participant'].astype(str)

    # define state from previous and participant id
    participant_mod = int(df['participant'].iloc[0][-1])
    df['state'] = (df['previous'] - participant_mod + 1) % 5

    # keep only states 1-4 if state 0 is not of interest
    df = df[df['state'].isin([1, 2, 3, 4])].copy()

    # count missed trials
    n_missed = (df['response'] == 'missed').sum()
    num_missed.append(n_missed)

    # make sure correctness is numeric
    df['correctness'] = pd.to_numeric(df['correctness'], errors='coerce')

    # missed trials should count as incorrect
    df.loc[df['response'] == 'missed', 'correctness'] = 0

    # -------------------------
    # RT summary: exclude missed
    # -------------------------
    df_rt = df[df['response'] != 'missed'][['participant', 'state', 'rt']].copy()
    df_rt = df_rt.groupby(['participant', 'state'], as_index=False).mean()

    # -------------------------
    # Correctness summary: include missed as 0
    # -------------------------
    df_corr = df[['participant', 'state', 'correctness']].copy()
    df_corr = df_corr.groupby(['participant', 'state'], as_index=False).mean()

    # merge participant-state summaries
    df_sub = pd.merge(df_rt, df_corr, on=['participant', 'state'], how='outer')
    dfs.append(df_sub)

df_all = pd.concat(dfs, ignore_index=True)

# =============================================================================
# Basic checks
# =============================================================================

print('\n=== NUMBER OF STATES PER PARTICIPANT ===')
print(df_all.groupby('participant')['state'].nunique().value_counts())

# keep only participants with all four states, for repeated-measures analyses
valid_participants = (
    df_all.groupby('participant')['state']
    .nunique()
    .loc[lambda x: x == 4]
    .index
)

df_all = df_all[df_all['participant'].isin(valid_participants)].copy()

print('\n=== INCLUDED PARTICIPANTS ===')
print(len(valid_participants))

# =============================================================================
# Plot RT by state
# =============================================================================

plt.figure(figsize=(8, 6))
sns.boxplot(x='state', y='rt', data=df_all)
sns.swarmplot(x='state', y='rt', data=df_all, color=".25")
plt.title('RT by state')
plt.tight_layout()
plt.show()

# =============================================================================
# Pairwise paired t-tests for RT
# =============================================================================

def paired_state_test(data, state_a, state_b, dv='rt'):
    pivot = data.pivot(index='participant', columns='state', values=dv)
    subset = pivot[[state_a, state_b]].dropna()
    return ttest_rel(subset[state_a], subset[state_b])

print('\n=== PAIRED T-TESTS: RT BETWEEN STATES ===')

print('test between 1 and 2')
print(paired_state_test(df_all, 1, 2, dv='rt'))

print('test between 1 and 3')
print(paired_state_test(df_all, 1, 3, dv='rt'))

print('test between 1 and 4')
print(paired_state_test(df_all, 1, 4, dv='rt'))

print('test between 2 and 3')
print(paired_state_test(df_all, 2, 3, dv='rt'))

print('test between 2 and 4')
print(paired_state_test(df_all, 2, 4, dv='rt'))

print('test between 3 and 4')
print(paired_state_test(df_all, 3, 4, dv='rt'))

# =============================================================================
# Descriptives for RT
# =============================================================================

for state in [1, 2, 3, 4]:
    print(f'\nmean and std for state {state} (rt)')
    print(df_all[df_all['state'] == state]['rt'].mean())
    print(df_all[df_all['state'] == state]['rt'].std())

print('\nmean across all states (rt)')
print(df_all['rt'].mean())
print(df_all['rt'].std())

# =============================================================================
# Descriptives for correctness
# =============================================================================

for state in [1, 2, 3, 4]:
    print(f'\nmean and std for state {state} (correctness)')
    print(df_all[df_all['state'] == state]['correctness'].mean())
    print(df_all[df_all['state'] == state]['correctness'].std())

print('\nmean across all states (correctness)')
print(df_all['correctness'].mean())
print(df_all['correctness'].std())

# =============================================================================
# Missed trials summary
# =============================================================================

print('\nmean and std for number of missed trials')
print(np.mean(num_missed))
print(np.std(num_missed))

# =============================================================================
# Repeated-measures ANOVAs
# =============================================================================

print('\n=== REPEATED-MEASURES ANOVA: RT ~ STATE ===')
aov_rt = AnovaRM(
    data=df_all,
    depvar='rt',
    subject='participant',
    within=['state']
)
res_rt = aov_rt.fit()
print(res_rt)

print('\n=== REPEATED-MEASURES ANOVA: CORRECTNESS ~ STATE ===')
aov_corr = AnovaRM(
    data=df_all,
    depvar='correctness',
    subject='participant',
    within=['state']
)
res_corr = aov_corr.fit()
print(res_corr)

# =============================================================================
# Save data
# =============================================================================

df_all.to_csv('/data/pt_02747/action_hippo/figures/rt_data.csv', index=False)