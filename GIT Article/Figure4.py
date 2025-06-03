import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

# === Helper for stacked‐bar + χ² annotation ===
def plot_stacked_crosstab(df, factor, category_col, ax, title):
    """
    Draw a stacked bar of df[factor] vs df[category_col],
    compute chi² test of independence, annotate p‐value.
    """
    # drop missing values
    df_tmp = df[[factor, category_col]].dropna()
    ct = pd.crosstab(df_tmp[factor], df_tmp[category_col])

    ct.plot(kind='bar', stacked=True, ax=ax,
            edgecolor='black', linewidth=0.8)
    ax.set_title(title, fontsize=16, pad=12)
    ax.set_xlabel(factor, fontsize=14)
    ax.set_ylabel('Count', fontsize=14)
    ax.tick_params(axis='x', rotation=90, labelsize=12)
    ax.tick_params(axis='y', labelsize=12)

    # χ² test
    chi2, p, _, _ = chi2_contingency(ct)
    ax.text(0.95, 0.95, f'χ² p = {p:.2e}',
            transform=ax.transAxes,
            ha='right', va='top',
            fontsize=12,
            bbox=dict(facecolor='white', edgecolor='gray', pad=3))


def main():
    factors = ['Specimen', 'Source', 'Age_Group', 'Gender', 'Ethnicity']

    # === Part 1: C. difficile toxin profiles ===
    df_cd = pd.read_excel('CD-Toxins.xlsx', parse_dates=['Date-Collected'])
    df_cd['Age'] = pd.to_numeric(df_cd['Age'], errors='coerce')
    # decade bins
    df_cd['Age_Group'] = pd.cut(
        df_cd['Age'],
        bins=np.arange(0, 101, 10),
        labels=[f'{i}-{i+9}' for i in range(0, 100, 10)],
        include_lowest=True
    )
    # organism‐positive = 'A'
    df_cd_pos = df_cd[df_cd['Result'] == 'A']

    fig_cd, axes_cd = plt.subplots(
        2, 3, figsize=(24, 16), constrained_layout=True
    )
    axes_flat_cd = axes_cd.flatten()
    for ax, factor in zip(axes_flat_cd, factors):
        plot_stacked_crosstab(
            df=df_cd_pos,
            factor=factor,
            category_col='Toxins',
            ax=ax,
            title=f'C. difficile: {factor} vs Toxins'
        )
    axes_flat_cd[-1].axis('off')  # hide extra subplot
    fig_cd.suptitle(
        'Figure 4A – C. difficile Toxin Profiles by Key Factors',
        fontsize=18, y=1.02
    )
    fig_cd.savefig('Figure4_CD.png', dpi=300)
    plt.close(fig_cd)

    # === Part 2: E. coli / Shigella subtypes ===
    df_ec = pd.read_excel('Ecoli-Shigella-toxins.xlsx',
                          parse_dates=['Date-Collected'])
    df_ec['Age'] = pd.to_numeric(df_ec['Age'], errors='coerce')
    df_ec['Age_Group'] = pd.cut(
        df_ec['Age'],
        bins=np.arange(0, 101, 10),
        labels=[f'{i}-{i+9}' for i in range(0, 100, 10)],
        include_lowest=True
    )
    # **Filter by Subtype Result 'P'**, not the organism Result
    df_ec_pos = df_ec[df_ec['Subtype Result'] == 'P']

    fig_ec, axes_ec = plt.subplots(
        2, 3, figsize=(24, 16), constrained_layout=True
    )
    axes_flat_ec = axes_ec.flatten()
    for ax, factor in zip(axes_flat_ec, factors):
        plot_stacked_crosstab(
            df=df_ec_pos,
            factor=factor,
            category_col='SubType',  # values: rfbA, Shigatoxin, NONE
            ax=ax,
            title=f'E. coli/Shigella: {factor} vs SubType'
        )
    axes_flat_ec[-1].axis('off')
    fig_ec.suptitle(
        'Figure 4B – E. coli & Shigella Subtypes by Key Factors',
        fontsize=18, y=1.02
    )
    fig_ec.savefig('Figure4_EC.png', dpi=300)
    plt.close(fig_ec)


if __name__ == '__main__':
    main()
