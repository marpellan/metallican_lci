import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.colors import TwoSlopeNorm
from matplotlib import rcParams
import random


# Set global Matplotlib parameters
rcParams['pdf.fonttype'] = 42  # Ensure TrueType fonts are embedded
rcParams['ps.fonttype'] = 42
rcParams['font.family'] = 'arial'  # Use serif fonts like Times New Roman or Palatino
rcParams['font.size'] = 10
rcParams['axes.labelsize'] = 10
rcParams['legend.fontsize'] = 9
rcParams['xtick.labelsize'] = 9
rcParams['ytick.labelsize'] = 9
rcParams['axes.titlesize'] = 12


def plot_midpoint_contributions(df, category, threshold=0.05, combine_terms=True, save_path=None):
    """
    Plots a high‑resolution stacked bar chart of midpoint contributions for each commodity,
    with optional merging of 'long term'/'short term' into a single category and
    thresholding to group small contributors into 'Other'.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain a 'Commodity' column and one column per midpoint (fractions 0–1).
    category : str
        'Human health' or 'Ecosystem quality' (used for title).
    threshold : float
        Minimum fraction (0–1) for a category to be shown; others are grouped into 'Other'.
    combine_terms : bool
        If True, merges columns ending in 'long term' or 'short term' into a single prefix.

    Usage
    -----
    plot_midpoint_shares_grouped(df_hh, 'Human health', threshold=0.05)
    plot_midpoint_shares_grouped(df_eq, 'Ecosystem quality', threshold=0.05)
    """
    df_proc = df.copy()
    # 1) optionally combine long/short term
    if combine_terms:
        def simplify(col):
            parts = col.split(', ')
            if parts[-1].lower() in ('long term', 'short term'):
                parts = parts[:-1]
            return ', '.join(parts)
        # Group by simplified label
        label_map = {}
        for col in df_proc.columns:
            if col == 'Commodity':
                continue
            label_map.setdefault(simplify(col), []).append(col)
        df_grouped = pd.DataFrame({'Commodity': df_proc['Commodity']})
        for label, cols in label_map.items():
            df_grouped[label] = df_proc[cols].sum(axis=1)
    else:
        df_grouped = df_proc

    # 2) thresholding: keep categories with max >= threshold
    cats = [c for c in df_grouped.columns if c != 'Commodity']
    keep = [c for c in cats if df_grouped[c].max() >= threshold]
    drop = [c for c in cats if c not in keep]

    # Build final df and compute 'Other'
    df_final = pd.DataFrame({'Commodity': df_grouped['Commodity']})
    for c in keep:
        df_final[c] = df_grouped[c]
    if drop:
        df_final['Other'] = df_grouped[drop].sum(axis=1)
        plot_cats = keep + ['Other']
    else:
        plot_cats = keep

    # Sort categories by mean share descending (largest at bottom of stack)
    plot_cats = sorted(plot_cats, key=lambda x: df_final[x].mean(), reverse=True)

    # 3) plotting
    x = range(len(df_final))
    bottoms = [0] * len(df_final)

    plt.figure(figsize=(12, 6), dpi=300)
    for cat_label in plot_cats:
        vals = df_final[cat_label].fillna(0).values * 100  # to percent
        plt.bar(x, vals, bottom=bottoms, label=cat_label)
        bottoms = [b + v for b, v in zip(bottoms, vals)]

    plt.xticks(x, df_final['Commodity'], rotation=45, ha='right')
    plt.ylabel('Percentage contribution (%)')
    plt.title(f'{category} midpoint contributions (≥{threshold*100:.0f}% or combined)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    # Save the plot if a path is provided, otherwise display it
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Plot saved to {save_path}")
    else:
        plt.show()


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

def heatmap_db_comparison(df_ei, df_ri, title=None, save_path=None):
    """
    Plots a heatmap to show the percent differences in impacts between two databases.

    Parameters:
        df_ei (pd.DataFrame): DataFrame with 'Commodity' as a column and impact categories as other columns.
        df_ri (pd.DataFrame): DataFrame with 'Commodity' as a column and impact categories as other columns.
        title (str, optional): Title for the plot.
        save_path (str, optional): Path to save the figure.
    """
    # Ensure 'Commodity' is set as index in both dataframes
    df1 = df_ei.set_index('Commodity')
    df2 = df_ri.set_index('Commodity')

    # Align columns and index
    df1, df2 = df1.align(df2, join='inner', axis=1)
    df1, df2 = df1.align(df2, join='inner', axis=0)

    # Calculate percentage difference
    df_diff = ((df2 - df1) / df1) * 100

    # Prepare plot dimensions
    num_rows, num_cols = df_diff.shape
    fig, ax = plt.subplots(figsize=(num_cols * 2, num_rows * 0.6 + 2), dpi=300)

    # Colormap centering
    min_val, max_val = df_diff.values.min(), df_diff.values.max()
    if min_val > 0:
        abs_max = max(abs(min_val), abs(max_val))
        vmin, vmax = -abs_max, abs_max
    elif max_val < 0:
        abs_max = max(abs(min_val), abs(max_val))
        vmin, vmax = -abs_max, abs_max
    else:
        vmin, vmax = min_val, max_val

    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    cmap = plt.get_cmap("coolwarm")

    # Draw cells
    for i, commodity in enumerate(df_diff.index):
        for j, col in enumerate(df_diff.columns):
            val = df_diff.iloc[i, j]
            color = cmap(norm(val))
            ax.add_patch(plt.Rectangle((j, i), 1, 1, color=color))
            ax.text(j + 0.5, i + 0.5, f"{val:.1f}%",
                    ha="center", va="center",
                    color="white" if abs(val) > (vmax - vmin) * 0.15 else "black",
                    fontsize=8)

    # Ticks and labels
    ax.set_xticks(np.arange(num_cols) + 0.5)
    ax.set_xticklabels(df_diff.columns, rotation=45, ha='right', fontsize=9)
    ax.set_yticks(np.arange(num_rows) + 0.5)
    ax.set_yticklabels(df_diff.index, fontsize=9)

    # Grid lines
    for y in range(1, num_rows):
        ax.axhline(y, color='black', linewidth=0.5)

    ax.set_xlim(0, num_cols)
    ax.set_ylim(num_rows, 0)

    if title:
        plt.title(title, fontsize=14, pad=12)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()

#
# def plot_multilca_impacts(df, colors=None, save_path=None):
#     """
#     Visualize LCA impacts for each category with specified or random colors and save the plot if a path is provided.
#
#     Parameters:
#     - df (pd.DataFrame): DataFrame with raw materials as index and impact categories as columns.
#     - colors (list of str): List of colors for each impact category. If None, random colors will be used.
#     - save_path (str, optional): Path to save the plot image. If None, the plot will only display.
#     """
#     # Ensure 'Raw material' is a column by resetting the index
#     #df = df.reset_index()
#     #df.rename(columns={'index': 'Raw material'}, inplace=True)
#
#     # Extract impact categories as all columns except 'Raw material'
#     impact_categories = [col for col in df.columns if col != 'Commodity']
#
#     # Generate random colors if none are provided
#     if colors is None:
#         colors = ['#%06X' % random.randint(0, 0xFFFFFF) for _ in impact_categories]
#
#     # Check that the number of colors matches the number of impact categories
#     if len(colors) != len(impact_categories):
#         raise ValueError("The number of colors must match the number of impact categories.")
#
#     # Set up the figure with subplots
#     plt.figure(figsize=(14, 10))
#
#     # Create a bar plot for each impact category with specified or random colors
#     for i, (impact, color) in enumerate(zip(impact_categories, colors), 1):
#         plt.subplot(2, 2, i)
#         plt.bar(df["Commodity"], df[impact], color=color, label=impact)
#         plt.xticks(rotation=90)
#         plt.ylabel(impact)
#         plt.title(impact)
#         plt.tight_layout()
#
#     # Save the plot if a path is provided, otherwise display it
#     if save_path:
#         plt.savefig(save_path, bbox_inches="tight")
#         print(f"Plot saved to {save_path}")
#     else:
#         plt.show()
#
#
# def plot_multilca_impacts_multidb(specific_lca_results, db_colors=None, save_path=None):
#     """
#     Visualize LCA impacts for each raw material across multiple databases with specified or random colors
#     per database, and save the plot if a path is provided.
#
#     Parameters:
#     - specific_lca_results (dict): Dictionary where keys are database names, and values are DataFrames
#                                    with raw materials as index and impact categories as columns.
#     - db_colors (dict): Dictionary mapping each database to a color. If None, random colors will be assigned.
#     - save_path (str, optional): Path to save the plot image. If None, the plot will only display.
#     """
#     # Get impact categories from the first database DataFrame
#     first_db = next(iter(specific_lca_results.values()))
#     impact_categories = first_db.columns.tolist()
#
#     # Generate random colors for each database if none are provided
#     if db_colors is None:
#         db_colors = {db_name: '#%06X' % random.randint(0, 0xFFFFFF) for db_name in specific_lca_results}
#
#     # Check that we have a color for each database
#     if len(db_colors) != len(specific_lca_results):
#         raise ValueError("The number of colors must match the number of databases.")
#
#     # Set up figure and axes
#     num_materials = len(first_db)
#     fig, axs = plt.subplots(len(impact_categories), 1, figsize=(15, 5 * len(impact_categories)), sharex=True)
#
#     # Create a grouped bar plot for each impact category
#     for i, impact in enumerate(impact_categories):
#         ax = axs[i] if len(impact_categories) > 1 else axs
#
#         # Width and offsets for bars
#         width = 0.15
#         x = range(num_materials)
#
#         # Plot each database for the current impact category
#         for j, (db_name, df) in enumerate(specific_lca_results.items()):
#             # Reset index to get the raw material names as a column
#             df = df.reset_index()
#             ax.bar([pos + j * width for pos in x], df[impact], width=width, label=db_name, color=db_colors[db_name])
#
#         ax.set_ylabel(impact)
#         ax.set_title(f"Comparison of {impact} across databases", fontsize=18, fontweight="bold", pad=20)
#         #ax.legend(title="Database")
#         ax.set_xticks([pos + width * (len(specific_lca_results) - 1) / 2 for pos in x])
#         ax.set_xticklabels(df["index"], rotation=90)
#
#     # Adjust layout
#     plt.tight_layout()
#
#     # Save or show plot
#     if save_path:
#         plt.savefig(save_path, bbox_inches="tight")
#         print(f"Plot saved to {save_path}")
#     else:
#         plt.show()
#
#
# def plot_contribution_analysis(df, inventory_names, colors=None, save_path=None):
#     """
#     Plot contribution analysis for multiple inventories with consistent colors for impact categories
#     and opacity based on score magnitude, sorted by contribution. Each plot is saved separately.
#
#     Parameters:
#     - df (pd.DataFrame): DataFrame containing the contribution analysis data.
#     - inventory_names (list of str): List of inventory names to filter and plot.
#     - colors (list of str): List of colors for each impact category.
#     - save_dir (str): Directory to save the plot images. Default is "results".
#     """
#
#     df = df.reset_index()
#
#     # Loop through each inventory in the provided list
#     for inventory_name in inventory_names:
#         # Filter the DataFrame for the specified inventory
#         df_inventory = df[df['Inventory'] == inventory_name]
#
#         # List of unique impact categories for the inventory
#         impact_categories = df_inventory['Impact Category'].unique()
#
#         # Set up figure size based on the number of impact categories
#         fig, axes = plt.subplots(len(impact_categories), 1, figsize=(10, len(impact_categories) * 2), sharex=True)
#
#         # Create a color dictionary for consistent coloring across categories
#         color_dict = {impact: colors[i % len(colors)] for i, impact in enumerate(impact_categories)}
#
#         # Loop through each impact category and create a horizontal bar plot
#         for i, impact_category in enumerate(impact_categories):
#             # Filter data for each impact category and sort by percentage in descending order
#             df_impact = df_inventory[df_inventory['Impact Category'] == impact_category]
#             df_impact = df_impact.sort_values(by='percentage', ascending=False)
#
#             # Normalize scores for opacity scaling
#             max_score = df_impact['score'].max()
#             normalized_scores = df_impact['score'] / max_score
#
#             # Plot each reference product with varying opacity based on normalized score
#             for ref_product, percentage, score in zip(df_impact['reference product'],
#                                                       df_impact['percentage'],
#                                                       normalized_scores):
#                 axes[i].barh(ref_product, percentage, color=color_dict[impact_category],
#                              alpha=score, edgecolor='black')
#
#             # Set plot labels and title
#             axes[i].set_xlim(0, 100)
#             axes[i].invert_yaxis()  # To have the highest contributions on top
#             axes[i].set_title(f"{impact_category} - {inventory_name}", color=color_dict[impact_category])
#             axes[i].set_xlabel("Contribution (%)")
#             axes[i].set_ylabel("")
#
#         plt.tight_layout()
#
#         # Save each plot to the specified directory with a filename based on the inventory name
#         plt.savefig(f"{save_path}_{inventory_name}.png", bbox_inches="tight", dpi=300)
#         print(f"Plot saved to {save_path}")
#         plt.close(fig)
#
#
# import matplotlib.pyplot as plt
# import pandas as pd
#
#
# def plot_iwplus_contributions(df, threshold_percent=3, save_path_eco=None, save_path_hh=None):
#     """
#     Plot separate bar charts for ecosystem quality and human health impact contributions,
#     aggregating minor contributions into an "Other" category if they fall below a given threshold.
#
#     Parameters:
#     - df (pd.DataFrame): DataFrame with raw materials as index and impact contributions as columns.
#     - threshold_percent (float): Threshold below which contributions are aggregated into "Other".
#     - save_path_eco (str, optional): Path to save the ecosystem quality plot.
#     - save_path_hh (str, optional): Path to save the human health plot.
#
#     Returns:
#     - None: Displays the plots and optionally saves them.
#     """
#
#     # Identify columns for ecosystem quality and human health
#     ecosystem_quality_cols = [col for col in df.columns if
#                               col.endswith("(PDF.m2.yr)") and col != "Total ecosystem quality (PDF.m2.yr)"]
#     human_health_cols = [col for col in df.columns if
#                          col.endswith("(DALY)") and col != "Total human health (DALY)"]
#
#     # Calculate the % impact for each contributor
#     for col in ecosystem_quality_cols:
#         df[f"{col} (%)"] = (df[col] / df["Total ecosystem quality (PDF.m2.yr)"]) * 100
#
#     for col in human_health_cols:
#         df[f"{col} (%)"] = (df[col] / df["Total human health (DALY)"]) * 100
#
#     # Function to aggregate impact categories below the threshold into "Other"
#     def aggregate_impact(df, impact_cols, threshold):
#         df_impact = df[["Commodity"] + [f"{col} (%)" for col in impact_cols]].set_index("Commodity")
#
#         # Identify contributions below the threshold
#         mask = df_impact < threshold
#         other_contribution = df_impact.where(mask).sum(axis=1)  # Sum contributions below threshold
#         df_impact = df_impact.where(~mask, 0)  # Set small contributors to 0
#         df_impact["Other (%)"] = other_contribution  # Add 'Other' category
#
#         # Drop any columns that are now completely 0
#         df_impact = df_impact.loc[:, (df_impact != 0).any(axis=0)]
#
#         return df_impact
#
#     df_ecosystem_quality_plot = aggregate_impact(df, ecosystem_quality_cols, threshold_percent)
#     df_human_health_plot = aggregate_impact(df, human_health_cols, threshold_percent)
#
#     # Plot for ecosystem quality contributions
#     fig_eco, ax_eco = plt.subplots(figsize=(18, 12))
#     df_ecosystem_quality_plot.plot(kind='bar', stacked=True, colormap='tab20', ax=ax_eco, legend=False)
#     ax_eco.set_title("Ecosystem Quality Impact", fontsize=18, fontweight="bold", pad=20)
#     ax_eco.set_ylabel("Impact Contribution (%)", fontsize=14, fontweight="bold", labelpad=10)
#     ax_eco.set_xlabel("", fontsize=14, fontweight="bold")
#     ax_eco.set_ylim(0, 100)
#     ax_eco.tick_params(axis='x', labelsize=16)
#
#     # Regenerate legend with only the displayed categories
#     handles_eco, labels_eco = ax_eco.get_legend_handles_labels()
#     labels_eco = df_ecosystem_quality_plot.columns.tolist()  # Update legend labels
#     fig_eco.legend(handles_eco, labels_eco, loc='upper center', ncol=4, bbox_to_anchor=(0.5, -0.05), fontsize=14)
#
#     plt.tight_layout()
#     if save_path_eco:
#         fig_eco.savefig(save_path_eco, bbox_inches='tight')
#     plt.show()
#
#     # Plot for human health contributions
#     fig_hh, ax_hh = plt.subplots(figsize=(18, 12))
#     df_human_health_plot.plot(kind='bar', stacked=True, colormap='tab20', ax=ax_hh, legend=False)
#     ax_hh.set_title("Human Health Impact", fontsize=18, fontweight="bold", pad=20)
#     ax_hh.set_ylabel("Impact Contribution (%)", fontsize=14, fontweight="bold", labelpad=10)
#     ax_hh.set_xlabel("", fontsize=14, fontweight="bold")
#     ax_hh.set_ylim(0, 100)
#     ax_hh.tick_params(axis='x', labelsize=16)
#
#     # Regenerate legend with only the displayed categories
#     handles_hh, labels_hh = ax_hh.get_legend_handles_labels()
#     labels_hh = df_human_health_plot.columns.tolist()  # Update legend labels
#     fig_hh.legend(handles_hh, labels_hh, loc='upper center', ncol=4, bbox_to_anchor=(0.5, -0.05), fontsize=14)
#
#     plt.tight_layout()
#     if save_path_hh:
#         fig_hh.savefig(save_path_hh, bbox_inches='tight')
#     plt.show()
#
#
# def heatmap_lca(df, colors=None, title=None, save_path=None):
#     """
#     Plots a heatmap with individually scaled color maps for each column and adds horizontal lines to separate rows.
#
#     Parameters:
#         df (pd.DataFrame): DataFrame with 'Commodity' as index and impact categories as columns.
#         colors (list): Optional. List of colormap names for each column.
#
#     Returns:
#         None. Displays the heatmap.
#     """
#     # Ensure 'Commodity' is the index and verify the order
#     if df.index.name != 'Commodity':
#         df = df.set_index('Commodity')
#
#     # Set default colors if none are provided
#     if colors is None:
#         colors = ['Greys', 'Oranges', 'Purples', 'Greens', 'Reds', 'Blues'][:len(df.columns)]
#
#     num_rows, num_cols = df.shape
#     fig, ax = plt.subplots(figsize=(num_cols * 2, num_rows * 0.5 + 2), dpi=300)  # Increase DPI for quality
#
#     # Create an array to hold colors
#     color_matrix = np.zeros((num_rows, num_cols, 4))  # RGBA colors
#
#     # For each column, map data to colors using its own colormap
#     for col_idx, (column, cmap_name) in enumerate(zip(df.columns, colors)):
#         col_values = df[column].values
#         norm = Normalize(vmin=col_values.min(), vmax=col_values.max())
#         cmap = plt.get_cmap(cmap_name)
#         # Map normalized data to colors with alpha channel
#         color_matrix[:, col_idx, :] = cmap(norm(col_values))  # RGBA values
#
#     # Expand color_matrix to match pcolormesh dimensions
#     color_matrix_expanded = np.zeros((num_rows + 1, num_cols + 1, 4))
#     color_matrix_expanded[:-1, :-1, :] = color_matrix
#
#     # Function to calculate luminance and determine text color
#     def get_text_color(rgba):
#         # Calculate relative luminance (perceived brightness)
#         r, g, b = rgba[:3]
#         luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
#         return "white" if luminance < 0.5 else "black"
#
#     # Create a grid of cells to color with pcolormesh
#     for i, commodity in enumerate(df.index):  # Ensure correct order by iterating directly over the DataFrame index
#         for j, column in enumerate(df.columns):
#             color = color_matrix_expanded[i, j]
#             ax.add_patch(plt.Rectangle((j, i), 1, 1, color=color))
#
#     # Set the ticks and labels
#     ax.set_xticks(np.arange(num_cols) + 0.5)
#     ax.set_yticks(np.arange(num_rows) + 0.5)
#     ax.set_xticklabels([f"{col}" for col in df.columns], rotation=45, ha='right', fontsize=14,
#                        fontweight='bold')
#     ax.set_yticklabels(df.index, fontsize=14, fontweight='bold')
#
#     # Loop over data dimensions and create text annotations
#     for i, commodity in enumerate(df.index):
#         for j, column in enumerate(df.columns):
#             value = df.loc[commodity, column]
#             # Format the number based on its magnitude
#             if abs(value) >= 1e6 or abs(value) <= 1e-2:
#                 text = f"{value:.2e}"  # Scientific notation
#             else:
#                 text = f"{value:.2f}"
#
#             # Determine appropriate text color based on background color luminance
#             color = color_matrix_expanded[i, j]
#             text_color = get_text_color(color)
#
#             ax.text(j + 0.5, i + 0.5, text, ha="center", va="center", color=text_color)
#
#     # Add horizontal lines between rows to separate commodities
#     ax.hlines(np.arange(1, num_rows), xmin=0, xmax=num_cols, color="black", linewidth=0.5)
#     ax.set_xlim(0, num_cols)
#     ax.set_ylim(num_rows, 0)  # Set the y-axis limits to prevent inversion
#
#     plt.title(title, fontsize=18, fontweight='bold', pad=20)
#     plt.tight_layout()
#     if save_path:
#         plt.savefig(save_path, dpi=300)
#     plt.show()
#
#
# def heatmap_db_comparison(df1, df2, title=None, save_path=None):
#     """
#     Plots a heatmap to show the differences in impacts between two databases.
#
#     Parameters:
#         df1 (pd.DataFrame): DataFrame with 'Commodity' as a column, and impact categories as other columns.
#         df2 (pd.DataFrame): DataFrame with 'Commodity' as a column, and impact categories as other columns.
#         title (str, optional): Title for the plot.
#         save_path (str, optional): Path to save the figure.
#
#     Returns:
#         None. Displays the heatmap.
#     """
#
#     # Ensure 'Commodity' is set as index in both dataframes
#     df1 = df1.set_index('Commodity')
#     df2 = df2.set_index('Commodity')
#
#     # Calculate percentage difference between the two DataFrames
#     df_diff = ((df2 - df1) / df1) * 100
#
#     # Determine figure size based on data dimensions
#     num_rows, num_cols = df_diff.shape
#     fig, ax = plt.subplots(figsize=(num_cols * 3, num_rows * 0.6 + 3), dpi=300)
#
#     # Extract min and max values
#     min_val = df_diff.values.min()
#     max_val = df_diff.values.max()
#
#     # Ensure zero is between vmin and vmax
#     # If all values are positive, or all are negative, we expand the range so 0 is included.
#     if min_val > 0:
#         # All values positive. Make vmin negative to include zero as center.
#         abs_max = max(abs(min_val), abs(max_val))
#         vmin, vmax = -abs_max, abs_max
#     elif max_val < 0:
#         # All values negative. Make vmax positive to include zero as center.
#         abs_max = max(abs(min_val), abs(max_val))
#         vmin, vmax = -abs_max, abs_max
#     else:
#         # Data crosses zero naturally
#         vmin, vmax = min_val, max_val
#
#     # Setup diverging colormap
#     cmap = plt.get_cmap("coolwarm")
#     norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
#
#     # Create the heatmap manually
#     for i, commodity in enumerate(df_diff.index):
#         for j, column in enumerate(df_diff.columns):
#             value = df_diff.loc[commodity, column]
#             color = cmap(norm(value))
#             ax.add_patch(plt.Rectangle((j, i), 1, 1, color=color))
#             # Display the percentage difference value
#             ax.text(j + 0.5, i + 0.5, f"{value:.1f}%", ha="center", va="center",
#                     color="white" if abs(value) > 50 else "black", fontsize=9)
#
#     # Set ticks and labels
#     ax.set_xticks(np.arange(num_cols) + 0.5)
#     ax.set_yticks(np.arange(num_rows) + 0.5)
#     ax.set_xticklabels(df_diff.columns, rotation=45, ha='right', fontsize=10, fontweight='bold')
#     ax.set_yticklabels(df_diff.index, fontsize=10, fontweight='bold')
#
#     # Add grid lines between rows
#     ax.hlines(np.arange(1, num_rows), xmin=0, xmax=num_cols, color="black", linewidth=0.5)
#
#     # Adjust axes limits
#     ax.set_xlim(0, num_cols)
#     ax.set_ylim(num_rows, 0)
#
#     # Add title if provided
#     if title:
#         plt.title(title, fontsize=18, fontweight='bold', pad=30)
#
#     # Adjust layout
#     plt.tight_layout(rect=[0, 0, 1, 0.95])
#
#     # Save figure if path is provided
#     if save_path:
#         plt.savefig(save_path, dpi=300, bbox_inches='tight')
#
#     # Show plot
#     plt.show()
#
#
# def pie_charts_technosphere_contribution(df, activity_col='Activity', method_col='LCIA Method',
#                                          flow_name_col='Flow Name', location_col='Flow Location',
#                                          value_col='Absolute Contribution', legend_size=10,
#                                          percentage_threshold=5, save_path=None):
#     """
#     Generate interactive pie charts for each combination of activity and LCIA method.
#     Contributions < percentage_threshold% are aggregated into an 'Other' category.
#
#     Parameters:
#     - df (pd.DataFrame): Input DataFrame with required columns.
#     - activity_col (str): Column name for Activity (e.g., minerals).
#     - method_col (str): Column name for LCIA Method.
#     - flow_name_col (str): Column name for Flow Name.
#     - location_col (str): Column name for Flow Location.
#     - value_col (str): Column name for contributions (e.g., Absolute Contribution).
#     - legend_size (int): Font size for the legend.
#     - percentage_threshold (float): Threshold percentage for aggregating into 'Other'.
#     - save_path (str): Path to save interactive HTML graphs (optional).
#     """
#     import plotly.express as px
#
#     # Merge 'Flow Name' and 'Flow Location' into a single label
#     df['Flow Label'] = df[flow_name_col] + ' (' + df[location_col] + ')'
#
#     # Get unique Activities and LCIA Methods
#     unique_activities = df[activity_col].unique()
#     unique_methods = df[method_col].unique()
#
#     # Loop through each activity and LCIA Method
#     for activity in unique_activities:
#         for method in unique_methods:
#             subset = df[(df[activity_col] == activity) & (df[method_col] == method)]
#
#             if subset.empty:
#                 continue
#
#             # Calculate total contribution and percentages
#             total_contribution = subset[value_col].sum()
#             subset['Percentage'] = (subset[value_col] / total_contribution) * 100
#
#             # Aggregate small contributions into 'Other'
#             above_threshold = subset[subset['Percentage'] >= percentage_threshold]
#             below_threshold = subset[subset['Percentage'] < percentage_threshold]
#
#             if not below_threshold.empty:
#                 other_sum = below_threshold[value_col].sum()
#                 other_row = pd.DataFrame({
#                     'Flow Label': ['Other'],
#                     value_col: [other_sum],
#                     'Percentage': [(other_sum / total_contribution) * 100]
#                 })
#                 subset_cleaned = pd.concat([above_threshold, other_row], ignore_index=True)
#             else:
#                 subset_cleaned = above_threshold
#
#             # Create interactive pie chart
#             fig = px.pie(
#                 subset_cleaned,
#                 names='Flow Label',
#                 values=value_col,
#                 title=f'{activity} - {method}',
#             )
#
#             # Update legend size
#             fig.update_layout(
#                 legend=dict(font=dict(size=legend_size)),
#                 title=dict(font=dict(size=14)),
#                 font=dict(family="Arial", size=12, color="black"),
#                 paper_bgcolor="white",
#                 plot_bgcolor="white"
#             )
#
#             # Show plot
#             #fig.show()
#
#             # Save as HTML if a path is provided
#             if save_path:
#                 filename = f"{save_path}/{activity}_{method}_pie_chart.html".replace(' ', '_')
#                 fig.write_html(filename)
#                 print(f"Saved: {filename}")
#
#
# ### Demand-related LCA ###
# def plot_scenario_production_comparison(df1, df2, save_path=None):
#     # Plotting with unified colors and separate legends for line styles
#     plt.figure(figsize=(14, 8))
#
#     # Unique minerals for color coding
#     unique_minerals = df1['Commodity'].unique()
#     colors = plt.cm.get_cmap('tab20', len(unique_minerals)).colors
#
#     # Plot each mineral with the same color for both dataframes
#     for i, mineral in enumerate(unique_minerals):
#         # Filter data for each mineral
#         df1_mineral = df1[df1['Commodity'] == mineral]
#         df2_mineral = df2[df2['Commodity'] == mineral]
#
#         # Plotting solid line for df1
#         plt.plot(df1_mineral['Year'], df1_mineral['Value'], label=f"{mineral}", color=colors[i])
#
#         # Plotting dashed line for df2
#         plt.plot(df2_mineral['Year'], df2_mineral['Value'], linestyle='--', color=colors[i])
#
#     # Adding titles and labels
#     plt.title("Existing production and production potential scenarios")
#     plt.xlabel("")
#     plt.ylabel("Production (kilotonnes)")
#
#     # Create custom legends for line types
#     solid_line = plt.Line2D([0], [0], color='black', linewidth=4, label='Existing production (df1)')
#     dashed_line = plt.Line2D([0], [0], color='black', linewidth=4, linestyle='--', label='Production potential (df2)')
#     plt.legend(handles=[solid_line, dashed_line], loc='upper left')
#
#     # Adding mineral legend separately
#     plt.legend(title="", loc='upper left')
#
#     # Save the figure if a path is provided
#     if save_path:
#         plt.savefig(save_path, format='png', dpi=300, bbox_inches='tight')
#
#     # Show plot
#     plt.show()
#
#
# def plot_production_impacts(projected_df, production_df,
#                                  impact_categories,
#                                  save_dir="results/demand_lca_results",
#                                  scenario_name=None,
#                                  lci_used=None):
#     '''
#     Function to plot projected impacts for each impact category as stacked bars with production volume as lines.
#
#     :param projected_df: DataFrame with projected impact data
#     :param production_df: DataFrame with production volume data
#     :param impact_categories: List of impact categories to plot
#     :param save_dir: Directory to save the plot image
#     :param scenario_name: Optional scenario name to include in the title and filenames
#     :return: None
#     '''
#
#     minerals = projected_df['Commodity'].unique()
#
#     # Set up colors for each mineral
#     colors = plt.cm.Set1.colors  # Use a colormap for consistent coloring
#
#     # Ensure the save directory exists
#     os.makedirs(save_dir, exist_ok=True)
#
#     # Create a figure with 4 subplots (2x2 layout)
#     fig, axs = plt.subplots(1, 2, figsize=(18, 8))
#     axs = axs.flatten()  # Flatten to easily iterate over for each category
#
#     for idx, category in enumerate(impact_categories):
#         ax1 = axs[idx]
#
#         # Prepare data for stacked bar chart
#         years = projected_df['Year'].unique()
#         bottom = np.zeros(len(years))  # Initialize bottom to zero for stacking
#
#         for i, mineral in enumerate(minerals):
#             subset = projected_df[projected_df['Commodity'] == mineral]
#             # Ensure all years are represented (fill missing years with zero)
#             values = [subset[subset['Year'] == year][category].values[0] if year in subset['Year'].values else 0 for
#                       year in years]
#             ax1.bar(years, values, bottom=bottom, color=colors[i % len(colors)], alpha=0.5, label=mineral)
#             bottom += values  # Update bottom for stacking next mineral
#
#         # Create a secondary y-axis for production volume
#         ax2 = ax1.twinx()
#
#         # Plot production volume as lines for each mineral
#         for i, mineral in enumerate(minerals):
#             production_subset = production_df[production_df['Commodity'] == mineral]
#             ax2.plot(production_subset['Year'], production_subset['Value'], linestyle='--',
#                      linewidth=2.5, color=colors[i % len(colors)], alpha=1)
#
#         # Set axis labels and title
#         #ax1.set_xlabel("Year")
#         #ax1.set_ylabel(f"{category} Impact")
#         ax2.set_ylabel("Production Volume (kilotonnes)")
#         ax1.set_title(f"{category} impact by mineral for {scenario_name}", pad=20, fontweight='bold', fontsize=14)
#
#     # Add legends outside the subplots
#     handles = [plt.Line2D([0], [0], color=color, lw=4) for color in colors[:len(minerals)]]
#     labels = minerals
#     fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.1), ncol=len(minerals), fontsize=18)
#
#     # Legend for the dashed production line
#     #fig.legend([plt.Line2D([0], [0], color='black', linestyle='--', lw=2, alpha=0.6)],
#     #               ['Production'], loc="lower center", bbox_to_anchor=(0.5, -0.1), fontsize=10, title="Line Style")
#
#     plt.tight_layout(rect=[0, 0, 1, 0.96])
#
#     # Save the combined figure
#     filename = f"combined_impact_{scenario_name}_{lci_used}.png"
#     save_path = os.path.join(save_dir, filename)
#     plt.savefig(save_path, bbox_inches="tight")
#     print(f"Combined plot saved to {save_path}")
#     plt.close(fig)
#
#
# def plot_incremental_impacts(projected_impacts_existing, projected_impacts_potential,
#                              production_existing, production_potential, impact_categories,
#                              save_dir="results/demand_lca_results", scenario_name="incremental"):
#     '''
#     Function to calculate and plot incremental impacts as stacked bars and production volumes as lines.
#
#     :param projected_impacts_existing: DataFrame with projected impact data for existing production scenario
#     :param projected_impacts_potential: DataFrame with projected impact data for potential production scenario
#     :param production_existing: DataFrame with production volume data for existing scenario
#     :param production_potential: DataFrame with production volume data for potential scenario
#     :param impact_categories: List of impact categories to plot
#     :param save_dir: Directory to save the plot image
#     :param scenario_name: Name to include in the filename
#     :return: None
#     '''
#
#     # Ensure the save directory exists
#     os.makedirs(save_dir, exist_ok=True)
#
#     # Calculate incremental impacts and production
#     impact_diff_df = pd.merge(projected_impacts_potential, projected_impacts_existing,
#                               on=['Year', 'Mineral'], suffixes=('_potential', '_existing'))
#     production_diff_df = pd.merge(production_potential, production_existing,
#                                   on=['Year', 'Mineral'], suffixes=('_potential', '_existing'))
#
#     # Calculate the difference for each impact category and production volume
#     for category in impact_categories:
#         impact_diff_df[f"{category}_diff"] = impact_diff_df[f"{category}_potential"] - impact_diff_df[
#             f"{category}_existing"]
#     production_diff_df["Value_diff"] = production_diff_df["Value_potential"] - production_diff_df["Value_existing"]
#
#     minerals = impact_diff_df['Mineral'].unique()
#
#     # Set up colors for each mineral
#     colors = plt.cm.Set1.colors  # Use a colormap for consistent coloring
#
#     # Create a figure with 4 subplots (2x2 layout)
#     fig, axs = plt.subplots(2, 2, figsize=(15, 12))
#     axs = axs.flatten()  # Flatten to easily iterate over for each category
#
#     for idx, category in enumerate(impact_categories):
#         ax1 = axs[idx]
#
#         # Prepare data for stacked bar chart
#         years = impact_diff_df['Year'].unique()
#         bottom = np.zeros(len(years))  # Track the bottom for stacking bars
#
#         for i, mineral in enumerate(minerals):
#             subset = impact_diff_df[impact_diff_df['Mineral'] == mineral]
#             # Align the data by year for stacking
#             values = [
#                 subset[subset['Year'] == year][f"{category}_diff"].values[0] if year in subset['Year'].values else 0 for
#                 year in years]
#             ax1.bar(years, values, bottom=bottom, color=colors[i % len(colors)], label=mineral)
#             bottom += values  # Update bottom for the next mineral layer
#
#         # Create a secondary y-axis for production volume (sum of minerals)
#         ax2 = ax1.twinx()
#
#         for i, mineral in enumerate(minerals):
#             production_subset = production_diff_df[production_diff_df['Mineral'] == mineral]
#             ax2.plot(production_subset['Year'], production_subset['Value_diff'],
#                      linestyle="--", color=colors[i % len(colors)], alpha=0.6)
#
#         # Set axis labels and title
#         ax1.set_xlabel("Year")
#         ax1.set_ylabel(f"{category} Incremental Impact")
#         ax2.set_ylabel("Incremental Production Volume (kilotonnes)")
#         ax1.set_title(f"{category} Incremental Impact from Potential Production")
#
#     # Add legends outside the subplots for stacked bars
#     fig.legend([plt.Line2D([0], [0], color=color, lw=2) for color in colors[:len(minerals)]],
#                minerals, loc="upper center", ncol=len(minerals), title="Minerals")
#     fig.legend([plt.Line2D([0], [0], color='black', linestyle="--", lw=2)],
#                ['Incremental Production'], loc="upper right", title="")
#
#     plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to make space for the legend
#
#     # Save the combined figure
#     filename = f"incremental_impact_{scenario_name}_stacked.png"
#     save_path = os.path.join(save_dir, filename)
#     plt.savefig(save_path, bbox_inches="tight")
#     print(f"Incremental impact stacked plot saved to {save_path}")
#     plt.close(fig)  # Close the figure to free up memory
