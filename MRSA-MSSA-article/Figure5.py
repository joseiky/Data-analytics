import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import fisher_exact
import networkx as nx
from upsetplot import from_indicators, UpSet

# === 1. Read full dataset ===
df = pd.read_excel('GIT-No-toxins.xlsx', parse_dates=['Date-Collected'])
df = df[df['Test'].notna()]

# Binarize presence per sample×pathogen
df['Presence'] = (df['Result'] == 'P').astype(int)
presence = (
    df.groupby(['MDLNo', 'Test'])['Presence']
      .max()
      .unstack(fill_value=0)
)

pathogens = presence.columns.tolist()
n = len(pathogens)

# === 2. Co-occurrence counts ===
co_matrix = presence.T.dot(presence)

# === 3. Fisher’s Exact Test for significance ===
pvals = pd.DataFrame(np.ones((n,n)), index=pathogens, columns=pathogens)
for i, p1 in enumerate(pathogens):
    for j, p2 in enumerate(pathogens):
        if j <= i:
            a = co_matrix.at[p1,p2]
            b = presence[p1].sum() - a
            c = presence[p2].sum() - a
            d = presence.shape[0] - (a + b + c)
            _, pv = fisher_exact([[a,b],[c,d]], alternative='greater')
            pvals.at[p1,p2] = pvals.at[p2,p1] = pv

alpha = 0.05 / (n*(n-1)/2)
signif = (pvals < alpha)

# === 4A. Heatmap ===
fig, ax = plt.subplots(figsize=(12,10))
cax = ax.matshow(co_matrix, cmap='Reds')
fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
ax.set_xticks(range(n)); ax.set_xticklabels(pathogens, rotation=90)
ax.set_yticks(range(n)); ax.set_yticklabels(pathogens)
for i in range(n):
    for j in range(n):
        ax.text(j, i, int(co_matrix.iat[i,j]),
                ha='center', va='center', fontsize=8)
ax.set_title('Figure 5A – Pathogen Co-occurrence Counts', pad=20, fontsize=16)
plt.tight_layout()
plt.savefig('Figure5A_Cooccurrence_Heatmap.png', dpi=300)
plt.close(fig)

# === 4B. Network graph ===
G = nx.Graph()
for p in pathogens:
    G.add_node(p, size=presence[p].sum())
for i, p1 in enumerate(pathogens):
    for p2 in pathogens[i+1:]:
        if signif.at[p1,p2]:
            G.add_edge(p1, p2, weight=co_matrix.at[p1,p2])

pos = nx.spring_layout(G, k=0.5, seed=42)
fig, ax = plt.subplots(figsize=(12,12))
nx.draw_networkx_nodes(
    G, pos,
    node_size=[G.nodes[p]['size']*2 for p in G.nodes()],
    node_color='skyblue',
    ax=ax
)
nx.draw_networkx_edges(
    G, pos,
    width=[G[u][v]['weight']/co_matrix.values.max()*5 for u,v in G.edges()],
    edge_color='gray',
    ax=ax
)
nx.draw_networkx_labels(G, pos, font_size=10, ax=ax)
ax.set_title('Figure 5B – Network of Significant Co-occurrences\n(α=0.05, Bonferroni)', fontsize=16)
ax.axis('off')
plt.tight_layout()
plt.savefig('Figure5B_Cooccurrence_Network.png', dpi=300)
plt.close(fig)

# === 4C. Upset plot ===
# Convert presence to boolean
presence_bool = presence.astype(bool)

# Build upset data using explicit indicator list
upset_data = from_indicators(pathogens, presence_bool)

fig = plt.figure(figsize=(10,6))
UpSet(upset_data, show_counts=True, sort_by='degree').plot(fig=fig)
plt.suptitle('Figure 5C – Upset Plot of Pathogen Co-detections', fontsize=16, y=1.02)
plt.savefig('Figure5C_Upset.png', dpi=300)
plt.close(fig)

print("Figures 5A, 5B & 5C generated (full dataset).")
