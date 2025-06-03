import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chisquare

# Path to your full dataset
DATA_PATH = 'GIT-No-toxins.xlsx'

def main():
    # 1. Read and parse dates
    df = pd.read_excel(DATA_PATH)
    df['Date-Collected'] = pd.to_datetime(df['Date-Collected'], errors='coerce')

    # 2. Filter to positives
    df_pos = df[df['Result'] == 'P'].copy()

    # 3. Overall counts
    pathogen_counts = df_pos['Test'].value_counts()

    # 4. Top-6 pathogens for temporal trends
    top6 = pathogen_counts.nlargest(7).index.tolist()

    # 5. Month-year periods
    df_pos['Period'] = df_pos['Date-Collected'].dt.to_period('M').astype(str)
    all_periods = sorted(df_pos['Period'].dropna().unique())

    # —— Create a 2×1 grid: bar chart on top, time series below —— #
    fig = plt.figure(figsize=(36, 16), constrained_layout=False)

    # 5 mm gap between panels
    gap_inches = 380 / 25.4
    fig_height = fig.get_size_inches()[1]
    hspace = gap_inches / fig_height
    fig.subplots_adjust(hspace=hspace)

    # Height ratio so top is slightly taller
    gs = fig.add_gridspec(2, 1, height_ratios=[1.5, 2])
    ax1 = fig.add_subplot(gs[0, 0])  # Figure 3A
    ax2 = fig.add_subplot(gs[1, 0])  # Figure 3B

    # —— Figure 3A: Pathogen frequencies —— #
    bars = ax1.bar(
        pathogen_counts.index,
        pathogen_counts.values,
        edgecolor='black',
        linewidth=1.2
    )
    ax1.set_title('Figure 3A – Pathogen Frequencies', fontsize=20, pad=16)
    ax1.set_xlabel('Pathogen', fontsize=16)
    ax1.set_ylabel('Positive Sample Count', fontsize=16)
    ax1.tick_params(axis='x', rotation=90, labelsize=14)
    ax1.tick_params(axis='y', labelsize=14)
    ax1.grid(axis='y', linestyle='--', alpha=0.5)

    max_count = pathogen_counts.max()
    for bar in bars:
        h = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            h + max_count * 0.01,
            f'{int(h)}',
            ha='center',
            va='bottom',
            fontsize=14
        )

    # —— Figure 3B: Seasonal trends —— #
    for pathogen in top6:
        ts = (
            df_pos[df_pos['Test'] == pathogen]
            .groupby('Period')
            .size()
            .reindex(all_periods, fill_value=0)
        )
        expected = [ts.mean()] * len(ts)
        chi2, pval = chisquare(ts.values, f_exp=expected)
        label = f'{pathogen} (χ² p={pval:.2e})'
        ax2.plot(
            all_periods,
            ts.values,
            marker='o',
            linewidth=2.5,
            label=label
        )

    ax2.set_xlabel('Month-Year', fontsize=16)
    ax2.set_ylabel('Positive Count', fontsize=16)
    ax2.tick_params(axis='x', rotation=90, labelsize=14)
    ax2.tick_params(axis='y', labelsize=14)
    ax2.grid(True, linestyle='--', alpha=0.5)

    # Restore centered title for Figure 3B (same style as 3A)
    ax2.set_title(
        'Figure 3B – Seasonal Trends of Top-6 Pathogens',
        fontsize=20,
        pad=16,
        loc='center'
    )

    # Legend inside top-right of the 3B plot
    ax2.legend(
        loc='upper right',
        bbox_to_anchor=(0.98, 0.98),
        ncol=1,
        fontsize=14,
        title='Pathogen (χ² p-value)',
        title_fontsize=16,
        frameon=False
    )

    # Save and show
    fig.savefig('Figure3_and_Seasonal_Supplement.png', dpi=300)
    plt.show()

if __name__ == '__main__':
    main()
