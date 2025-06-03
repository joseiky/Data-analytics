import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import f_oneway

# Path to your full dataset
DATA_PATH = 'GIT-No-toxins.xlsx'

def boxplot_with_anova(df, group_col, value_col, ax, title):
    """
    Create a boxplot of value_col grouped by group_col on ax,
    and compute an ANOVA p-value annotation.
    """
    # Gather groups for ANOVA
    groups = [grp[value_col].dropna() for _, grp in df.groupby(group_col)]
    if len(groups) > 1:
        fstat, pval = f_oneway(*groups)
        p_text = f'ANOVA p = {pval:.2e}'
    else:
        p_text = 'ANOVA p = N/A'
    # Plot
    df.boxplot(column=value_col, by=group_col, ax=ax, grid=False,
               boxprops=dict(linewidth=1.5),
               medianprops=dict(linewidth=1.5, color='firebrick'))
    ax.set_title(f'{title}\n{p_text}', fontsize=12)
    ax.set_xlabel(group_col, fontsize=10)
    ax.set_ylabel(value_col, fontsize=10)
    ax.tick_params(axis='x', rotation=90)
    plt.suptitle('')  # remove default title

def main():
    # Read the entire dataset
    df = pd.read_excel(DATA_PATH)

    # Ensure Age is numeric
    df['Age'] = pd.to_numeric(df['Age'], errors='coerce')

    # Create figure with up to 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)

    # 1) Age by Pathogen (Test)
    boxplot_with_anova(df, 'Test', 'Age', axes[0], 'Age by Pathogen')

    # 2) Age by Ethnicity
    boxplot_with_anova(df, 'Ethnicity', 'Age', axes[1], 'Age by Ethnicity')

    # 3) Age by Specimen Source
    boxplot_with_anova(df, 'Source', 'Age', axes[2], 'Age by Specimen Source')

    # Save high-res figure
    plt.savefig('age_distribution_boxplots.png', dpi=400)
    plt.show()

if __name__ == '__main__':
    main()
