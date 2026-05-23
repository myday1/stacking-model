import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator
import seaborn as sns

# Set academic style parameters
#plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Configure matplotlib for publication quality
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 14,
    'lines.linewidth': 1.5,
    'axes.linewidth': 1.2,
    'xtick.major.width': 1.2,
    'ytick.major.width': 1.2,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'figure.dpi': 300,
})


def create_shap_summary_plot(output_path='shap_summary.eps'):
    """
    Create academic-style SHAP summary plot in EPS format

    Parameters:
    -----------
    output_path : str
        Output file path (e.g., 'shap_summary.eps')
    """

    # Feature data with SHAP values
    features = [
        'PM2.5', 'AQI_roll_3day', 'O3_8h', 'PM10', 'AQI_lag_2',
        'AQI_lag_1', 'FAO_Precipitation', 'Transformer_OOF', 'AQI_lag_3',
        'SO2', 'Temperature_2m', 'CO', 'NO2', 'Relative_Humidity_2m',
        'Wind_Speed_10m', 'Wind_Direction_10m', 'Surface_Pressure', 'Month'
    ]

    # Mean absolute SHAP values
    shap_values = np.array([128, 105, 58, 45, 12, 8, 6, 5, 4, 4, 3, 3, 2, 2, 2, 1, 1, 1])

    # Feature categories for coloring
    categories = [
        'Pollutant', 'Temporal', 'Pollutant', 'Pollutant', 'Temporal',
        'Temporal', 'Weather', 'Model', 'Temporal', 'Pollutant',
        'Weather', 'Pollutant', 'Pollutant', 'Weather', 'Weather',
        'Weather', 'Weather', 'Temporal'
    ]

    # Color mapping
    color_map = {
        'Pollutant': '#e74c3c',
        'Weather': '#3498db',
        'Temporal': '#9b59b6',
        'Model': '#f39c12'
    }
    colors = [color_map[cat] for cat in categories]

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Sort by SHAP values
    sorted_indices = np.argsort(shap_values)
    sorted_features = [features[i] for i in sorted_indices]
    sorted_values = shap_values[sorted_indices]
    sorted_colors = [colors[i] for i in sorted_indices]

    # Create horizontal bar plot
    y_pos = np.arange(len(sorted_features))
    bars = ax.barh(y_pos, sorted_values, color=sorted_colors, alpha=0.8, edgecolor='black', linewidth=0.5)

    # Customize axes
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_features, fontsize=10, family='monospace')
    ax.set_xlabel('Mean |SHAP value| (Average impact on model output)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Features', fontsize=11, fontweight='bold')
    ax.set_title('SHAP Feature Importance Analysis\nAir Quality Prediction Model',
                 fontsize=13, fontweight='bold', pad=15)

    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, sorted_values)):
        ax.text(value + 2, bar.get_y() + bar.get_height() / 2,
                f'{int(value)}',
                va='center', ha='left', fontsize=8, fontweight='bold')

    # Grid
    ax.grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)

    # Add legend
    legend_elements = [
        mpatches.Patch(facecolor='#e74c3c', edgecolor='black', linewidth=0.5, label='Pollutant'),
        mpatches.Patch(facecolor='#3498db', edgecolor='black', linewidth=0.5, label='Meteorological'),
        mpatches.Patch(facecolor='#9b59b6', edgecolor='black', linewidth=0.5, label='Temporal/Lag'),
        mpatches.Patch(facecolor='#f39c12', edgecolor='black', linewidth=0.5, label='Model'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9, framealpha=0.95, edgecolor='black')

    # Set x-axis limits with padding
    ax.set_xlim(0, max(sorted_values) * 1.15)

    # Adjust layout
    plt.tight_layout()

    # Save as EPS
    plt.savefig(output_path, format='eps', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"Figure saved: {output_path}")

    plt.show()


def create_feature_category_plot(output_path='feature_categories.eps'):
    """
    Create academic-style feature category breakdown plot in EPS format

    Parameters:
    -----------
    output_path : str
        Output file path
    """

    categories = ['Pollutant', 'Temporal/Lag', 'Meteorological', 'Model']
    importance_sum = [229, 35, 18, 5]  # Sum of SHAP values per category
    colors = ['#e74c3c', '#9b59b6', '#3498db', '#f39c12']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Pie chart
    wedges, texts, autotexts = ax1.pie(importance_sum, labels=categories, autopct='%1.1f%%',
                                       colors=colors, startangle=90,
                                       textprops={'fontsize': 10, 'fontweight': 'bold'},
                                       wedgeprops={'edgecolor': 'black', 'linewidth': 1.5})
    ax1.set_title('Feature Category Distribution\n(by Total SHAP Impact)',
                  fontsize=12, fontweight='bold', pad=15)

    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(10)
        autotext.set_fontweight('bold')

    # Bar chart
    bars = ax2.bar(categories, importance_sum, color=colors, alpha=0.8,
                   edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('Cumulative SHAP Impact', fontsize=11, fontweight='bold')
    ax2.set_title('Feature Category Importance Ranking', fontsize=12, fontweight='bold', pad=15)
    ax2.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
    ax2.set_axisbelow(True)

    # Add value labels
    for bar, value in zip(bars, importance_sum):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2., height + 3,
                 f'{int(value)}',
                 ha='center', va='bottom', fontweight='bold', fontsize=10)

    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()

    # Save as EPS
    plt.savefig(output_path, format='eps', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"Figure saved: {output_path}")

    plt.show()


def create_detailed_feature_table(output_path='feature_details.eps'):
    """
    Create academic-style detailed feature table as EPS format

    Parameters:
    -----------
    output_path : str
        Output file path
    """

    features_info = [
        ('PM2.5', 'Fine particulate matter', 'Pollutant', 128),
        ('AQI_roll_3day', '3-day rolling average AQI', 'Temporal', 105),
        ('O3_8h', 'Ozone 8-hour average', 'Pollutant', 58),
        ('PM10', 'Inhalable particulate matter', 'Pollutant', 45),
        ('AQI_lag_2', 'Lagged AQI (2 periods)', 'Temporal', 12),
        ('AQI_lag_1', 'Lagged AQI (1 period)', 'Temporal', 8),
        ('FAO_Precipitation', 'Rainfall amount', 'Meteorological', 6),
        ('Transformer_OOF', 'Neural network predictions', 'Model', 5),
    ]

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.axis('tight')
    ax.axis('off')

    # Prepare table data
    table_data = [['Feature', 'Description', 'Category', 'SHAP Value']]
    for feat, desc, cat, val in features_info:
        table_data.append([feat, desc, cat, f'{val}'])

    # Create table
    table = ax.table(cellText=table_data, cellLoc='left', loc='center',
                     colWidths=[0.18, 0.38, 0.18, 0.15])

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.5)

    # Style header row
    for i in range(4):
        cell = table[(0, i)]
        cell.set_facecolor('#34495e')
        cell.set_text_props(weight='bold', color='white', fontsize=10)
        cell.set_edgecolor('black')
        cell.set_linewidth(1.5)

    # Style data rows with alternating colors
    color_map = {
        'Pollutant': '#fadbd8',
        'Meteorological': '#d6eaf8',
        'Temporal': '#ebdef0',
        'Model': '#fdebd0'
    }

    for i in range(1, len(table_data)):
        category = table_data[i][2]
        for j in range(4):
            cell = table[(i, j)]
            cell.set_facecolor(color_map.get(category, '#ffffff'))
            cell.set_edgecolor('black')
            cell.set_linewidth(0.8)
            if j == 0:  # Feature name column
                cell.set_text_props(family='monospace', weight='bold')

    plt.title('Top Features: Descriptions and SHAP Importance\nAir Quality Prediction Model',
              fontsize=13, fontweight='bold', pad=20)

    plt.savefig(output_path, format='eps', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"Figure saved: {output_path}")

    plt.show()


if __name__ == "__main__":
    # Generate all plots
    print("Generating academic-style SHAP plots in EPS format...\n")

    create_shap_summary_plot('shap_summary.eps')
    print()

    create_feature_category_plot('feature_categories.eps')
    print()

    create_detailed_feature_table('feature_details.eps')
    print()

    print("✓ All plots generated successfully!")
    print("\nOutput files:")
    print("  1. shap_summary.eps - Main SHAP feature importance plot")
    print("  2. feature_categories.eps - Category distribution analysis")
    print("  3. feature_details.eps - Detailed feature information table")