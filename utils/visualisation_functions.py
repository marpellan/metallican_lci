import os
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.colors import TwoSlopeNorm
from matplotlib import rcParams
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import random
from scipy.stats import linregress


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


def plot_2axes_by_commodity(
    df,
    x_col='commodities',
    y_col='energy_MJ',
    color_col='province',
    symbol_col='mining_processing_type',
    hover_name_cols=['facility_name', 'facility_group_name'],
    x_label=None,
    y_label=None,
    font_color="#333333",
    export_path=None,
    export_format="png",
    export_scale=3
):
    """
    Interactive scatter plot for energy by commodity.
    - Province = color
    - Mining/processing type = symbol
    - Separate legends for color and symbols
    - Hover shows original info (not marker_color/symbol)
    """

    df = df.copy()

    # Build hover_name
    if hover_name_cols and all(col in df.columns for col in hover_name_cols):
        df['hover_name'] = (
            df[hover_name_cols[0]].astype(str)
            + " (" + df[hover_name_cols[1]].astype(str) + ")"
        )
    else:
        df['hover_name'] = df[hover_name_cols[0]] if hover_name_cols else None

    # Unique categories
    unique_colors = df[color_col].dropna().unique()
    unique_symbols = df[symbol_col].dropna().unique()

    # Assign colors and symbols
    color_map = {col: px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)]
                 for i, col in enumerate(unique_colors)}

    symbol_list = [
        "circle", "square", "diamond", "cross", "x",
        "triangle-up", "triangle-down", "triangle-left", "triangle-right",
        "star", "hexagon", "pentagon"
    ]
    symbol_map = {sym: symbol_list[i % len(symbol_list)] for i, sym in enumerate(unique_symbols)}

    # Initialize figure
    fig = go.Figure()

    # Add scatter points for all data
    for color in unique_colors:
        for sym in unique_symbols:
            df_sub = df[(df[color_col] == color) & (df[symbol_col] == sym)]
            if df_sub.empty:
                continue
            fig.add_trace(
                go.Scatter(
                    x=df_sub[x_col],
                    y=df_sub[y_col],
                    mode='markers',
                    marker=dict(
                        size=10,
                        color=color_map[color],
                        symbol=symbol_map[sym],
                        line=dict(width=1, color='DarkSlateGrey')
                    ),
                    text=df_sub['hover_name'],
                    hovertemplate='%{text}<br>' + color_col + ': ' + color + '<br>' +
                                  symbol_col + ': ' + sym + '<br>' + y_col + ': %{y:.2f}<extra></extra>',
                    name=f"{color} - {sym}",
                    showlegend=False  # Legend handled separately
                )
            )

    # --- Color legend ---
    for color in unique_colors:
        fig.add_trace(
            go.Scatter(
                x=[None], y=[None],
                mode='markers',
                marker=dict(symbol='circle', size=10, color=color_map[color]),
                legendgroup='Color',
                showlegend=True,
                name=str(color)
            )
        )

    # --- Symbol legend ---
    for sym in unique_symbols:
        fig.add_trace(
            go.Scatter(
                x=[None], y=[None],
                mode='markers',
                marker=dict(symbol=symbol_map[sym], size=10, color='grey'),
                legendgroup='Symbol',
                showlegend=True,
                name=str(sym)
            )
        )

    # Layout
    fig.update_layout(
        font=dict(color=font_color, size=14),
        xaxis_title=x_label if x_label else x_col.replace('_', ' ').title(),
        yaxis_title=y_label if y_label else y_col.replace('_', ' ').title(),
        xaxis=dict(tickangle=90),
        legend=dict(tracegroupgap=20, itemsizing='constant'),
        template='plotly_white',
    )

    # Export
    if export_path:
        if export_format.lower() in ["png", "svg"]:
            fig.write_image(export_path, format=export_format, scale=export_scale)
        elif export_format.lower() == "html":
            fig.write_html(export_path, include_plotlyjs="cdn")
        else:
            raise ValueError("export_format must be 'png', 'svg', or 'html'")

    fig.show()
    return fig


def plot_biosphere(
    biosphere_df,
    x_col='commodities',
    y_col='value_normalized',
    color_col='province',
    symbol_col='mining_processing_type',
    hover_name_cols=['facility_name', 'facility_group_name'],
    y_unit_col='unit_normalized',
    save_path=None
):
    """
    Interactive scatter plot with dropdown to select substance_name.
    Color = province
    Symbol = mining/processing type
    Hover = facility_name (facility_group_name)
    Aggregates values per substance_name + main_id + color + symbol + hover_name + unit.
    """
    df = biosphere_df.copy()

    # Build hover_name
    if hover_name_cols and all(col in df.columns for col in hover_name_cols):
        df['hover_name'] = (
            df[hover_name_cols[0]].astype(str)
            + " (" + df[hover_name_cols[1]].astype(str) + ")"
        )
    else:
        df['hover_name'] = df[hover_name_cols[0]] if hover_name_cols else None

    # --- Aggregate values to avoid duplicate points ---
    agg_cols = ['substance_name', x_col, color_col, symbol_col, 'hover_name', y_unit_col]
    df_agg = df.groupby(agg_cols, as_index=False)[y_col].sum()

    substances = df_agg['substance_name'].unique()

    # Prepare color and symbol sequences
    color_sequence = px.colors.qualitative.Plotly
    symbol_sequence = [
        "circle", "square", "diamond", "cross", "x",
        "triangle-up", "triangle-down", "triangle-left", "triangle-right",
        "star", "hexagon", "pentagon"
    ]

    # Initialize figure
    fig = go.Figure()

    # --- Add traces per substance (only first visible) ---
    for i, substance in enumerate(substances):
        df_sub = df_agg[df_agg['substance_name'] == substance]
        visible = True if i == 0 else False

        fig.add_trace(
            go.Scatter(
                x=df_sub[x_col],
                y=df_sub[y_col],
                mode='markers',
                marker=dict(
                    size=10,
                    color=[color_sequence[j % len(color_sequence)] for j in range(len(df_sub))],
                    symbol=[symbol_sequence[j % len(symbol_sequence)] for j in range(len(df_sub))]
                ),
                text=df_sub['hover_name'],
                hovertemplate='%{text}<br>%{y:.10f} ' + df_sub[y_unit_col].iloc[0] + '<extra></extra>',
                name=substance,        # not shown in legend
                visible=visible,
                showlegend=False       # hide substance legend
            )
        )

    # --- Add dummy traces for color legend ---
    unique_colors = df_agg[color_col].dropna().unique()
    for i, col in enumerate(unique_colors):
        fig.add_trace(
            go.Scatter(
                x=[None], y=[None], mode='markers',
                marker=dict(symbol="circle", size=10, color=color_sequence[i % len(color_sequence)]),
                legendgroup="Color",
                showlegend=True,
                name=str(col)
            )
        )

    # --- Add dummy traces for symbol legend ---
    unique_symbols = df_agg[symbol_col].dropna().unique()
    for i, sym in enumerate(unique_symbols):
        fig.add_trace(
            go.Scatter(
                x=[None], y=[None], mode='markers',
                marker=dict(symbol=symbol_sequence[i % len(symbol_sequence)], size=10, color="grey"),
                legendgroup="Symbol",
                showlegend=True,
                name=str(sym)
            )
        )

    # --- Dropdown buttons ---
    buttons = []
    for i, substance in enumerate(substances):
        visibility = [False]*len(substances) + [True]*(len(unique_colors)+len(unique_symbols))
        visibility[i] = True
        y_unit = df_agg[df_agg['substance_name']==substance][y_unit_col].iloc[0]
        buttons.append(
            dict(
                label=substance,
                method="update",
                args=[{"visible": visibility},
                      {"title": f"Impacts for {substance}",
                       "yaxis": {"title": f"Impact ({y_unit})"}}]
            )
        )

    fig.update_layout(
        updatemenus=[dict(buttons=buttons, direction="down", showactive=True)],
        title=f"Impacts for {substances[0]}",
        xaxis_title=x_col.replace('_', ' ').title(),
        yaxis_title=f"Impact ({df_agg[df_agg['substance_name']==substances[0]][y_unit_col].iloc[0]})",
        xaxis=dict(tickangle=45),
        height=600,
        template="plotly_white",
        legend=dict(tracegroupgap=20, itemsizing='constant')
    )

    if save_path:
        fig.write_html(save_path, include_plotlyjs='cdn')

    fig.show()
    return fig


def plot_stacked_energy_by_site(
    df,
    x_col='facility_name',  # or a combined site-commodity column
    y_col='value_normalized_sum',
    color_col='subflow_type_agg',  # e.g., 'electricity', 'diesel', etc.
    symbol_col='',  # optional, for legend
    hover_name_cols=['facility_name', 'commodity'],
    x_label=None,
    y_label='MJ/t ore processed',
    font_color="#333333",
    export_path=None,
    export_format="html",
    export_scale=3
):
    """
    Stacked bar plot for energy types by site.
    - Site (and commodity) = x-axis
    - Energy type = color (stacked)
    - Mining/processing type = legend (optional)
    - Hover shows site, commodity, and energy breakdown
    """
    df = df.copy()
    # Build hover_name
    if hover_name_cols and all(col in df.columns for col in hover_name_cols):
        df['hover_name'] = (
            df[hover_name_cols[0]].astype(str)
            + " (" + df[hover_name_cols[1]].astype(str) + ")"
        )
    else:
        df['hover_name'] = df[hover_name_cols[0]] if hover_name_cols else None

    # Create the stacked bar plot
    fig = px.bar(
        df,
        x=x_col,
        y=y_col,
        color=color_col,
        barmode='stack',
        title="",
        labels={x_col: x_label if x_label else x_col.replace('_', ' ').title(),
                y_col: y_label if y_label else y_col.replace('_', ' ').title()},
        hover_data=['hover_name'],
        hover_name='hover_name',
    )

    # Customize layout
    fig.update_layout(
        font=dict(color=font_color, size=14),
        xaxis_title=x_label if x_label else x_col.replace('_', ' ').title(),
        yaxis_title=y_label if y_label else y_col.replace('_', ' ').title(),
        xaxis=dict(tickangle=90),
        legend=dict(title=color_col.replace('_', ' ').title()),
        template='plotly_white',
    )

    # Optional: Add symbol legend (if needed)
    if symbol_col in df.columns:
        unique_symbols = df[symbol_col].dropna().unique()
        symbol_list = [
            "circle", "square", "diamond", "cross", "x",
            "triangle-up", "triangle-down", "triangle-left", "triangle-right",
            "star", "hexagon", "pentagon"
        ]
        symbol_map = {sym: symbol_list[i % len(symbol_list)] for i, sym in enumerate(unique_symbols)}
        for sym in unique_symbols:
            fig.add_trace(
                go.Scatter(
                    x=[None], y=[None],
                    mode='markers',
                    marker=dict(symbol=symbol_map[sym], size=10, color='grey'),
                    legendgroup='Symbol',
                    showlegend=True,
                    name=str(sym)
                )
            )

    # Export
    if export_path:
        if export_format.lower() in ["png", "svg"]:
            fig.write_image(export_path, format=export_format, scale=export_scale)
        elif export_format.lower() == "html":
            fig.write_html(export_path, include_plotlyjs="cdn")
        else:
            raise ValueError("export_format must be 'png', 'svg', or 'html'")

    fig.show()
    return fig


def plot_relative_difference_heatmap(df_ei, df_ri, title, output_png=None, output_pdf=None):
    """
    Génère une heatmap de la différence relative en pourcentage entre deux DataFrames.
    Rouge : Regionvent > Ecoinvent
    Bleu : Ecoinvent > Regionvent

    :param df_ei: DataFrame Ecoinvent.
    :param df_ri: DataFrame Regionvent.
    :param title: Titre de la heatmap.
    :param output_png: Chemin pour enregistrer le PNG (optionnel).
    :param output_pdf: Chemin pour enregistrer le PDF (optionnel).
    """
    # S'assurer que les colonnes sont les mêmes
    common_columns = df_ei.columns.intersection(df_ri.columns)
    df_ei = df_ei[common_columns]
    df_ri = df_ri[common_columns]

    # Calculer la différence relative en pourcentage
    difference = ((df_ri.set_index('Commodity') - df_ei.set_index('Commodity')) / df_ri.set_index('Commodity')) * 100

    # Générer la heatmap avec une palette plus contrastée
    plt.figure(figsize=(16, 12))
    sns.set(font_scale=1.2)
    ax = sns.heatmap(difference, cmap='RdBu_r', center=0, annot=False, fmt=".1f",
                     linewidths=.5, cbar_kws={'label': 'Relative Difference with Ecoinvent values (%)'})


    plt.title(title, fontsize=16)
    plt.xticks(rotation=45, ha='right', fontsize=16)
    ax.set_ylabel('')
    plt.yticks(fontsize=14)
    plt.tight_layout()

    # Enregistrer si des chemins sont fournis
    if output_png:
        plt.savefig(output_png, dpi=300, bbox_inches='tight')
    if output_pdf:
        plt.savefig(output_pdf, bbox_inches='tight')

    plt.show()


def create_sankey_diagram(data):
    """
    Generates a Sankey diagram from the provided DataFrame.

    Args:
        data (pd.DataFrame): A DataFrame containing the raw data.

    Returns:
        plotly.graph_objects.Figure: The generated Sankey figure.
    """
    # Create a copy to avoid modifying the original DataFrame
    df = data.copy()

    # Data cleaning and preparation
    # 1. Clean the 'commodities' column to handle different orderings of the same items.
    #    This ensures "Gold, copper, silver" and "Copper, silver, gold" are treated the same.
    df['commodities_cleaned'] = df['commodities'].apply(
        lambda x: ', '.join(sorted([c.strip().lower() for c in str(x).split(',')])) if pd.notna(x) else 'Unknown Commodities'
    )

    # 2. Define the columns to be used for the Sankey flow
    columns = [
        'commodities_cleaned',
        'province',
        'mining_processing_type',
        'mining_method',
        'mining_submethod'
    ]

    # Replace NaN values in the flow columns with a specific 'Unknown' label for each column
    for col in columns:
        # We handle commodities_cleaned separately, so we skip it here.
        if col != 'commodities_cleaned':
            df[col] = df[col].fillna(f'Unknown {col.replace("_", " ").title()}')

    # 3. Apply custom prefixes to ensure uniqueness and readability
    df['mining_processing_type'] = df['mining_processing_type'].apply(lambda x: f"NRC: {x}")
    df['mining_method'] = df['mining_method'].apply(lambda x: f"MDO: {x}")
    df['mining_submethod'] = df['mining_submethod'].apply(lambda x: f"MDO_s: {x}")

    # Create a list of all unique labels for the nodes
    labels = []
    for col in columns:
        labels.extend(df[col].unique().tolist())

    labels = sorted(list(set(labels)))

    # Create a mapping from label to an index for Plotly
    label_to_index = {label: i for i, label in enumerate(labels)}

    # Initialize lists to store the source, target, and value of each link
    source = []
    target = []
    value = []

    # Iterate through the defined flow path and create the links
    for i in range(len(columns) - 1):
        col1 = columns[i]
        col2 = columns[i+1]

        # Group by the source and target nodes to count the links
        link_counts = df.groupby([col1, col2]).size().reset_index(name='count')

        for _, row in link_counts.iterrows():
            source_label = row[col1]
            target_label = row[col2]
            link_value = row['count']

            # Ensure both source and target labels exist in our map
            if pd.notna(source_label) and pd.notna(target_label):
                source.append(label_to_index[source_label])
                target.append(label_to_index[target_label])
                value.append(link_value)

    # Create the Sankey diagram using plotly
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=25,  # Increased pad for vertical layout
            thickness=25, # Increased thickness for vertical layout
            line=dict(color="black", width=0.5),
            label=labels,
            hovertemplate='%{label}<br>Value: %{value} facilities<extra></extra>'
        ),
        link=dict(
            source=source,
            target=target,
            value=value,
            hovertemplate='From %{source.label}<br>To %{target.label}<br>Count: %{value}<extra></extra>'
        ),
        orientation='v' # Set the orientation to vertical here
    )])

    # Update layout for a better visualization
    fig.update_layout(
        title_text="Sankey Diagram of Mining Facilities",
        font=dict(size=10),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )

    return fig


import os
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_contribution_subplots(df, value_col="Share_%", label_col="Activity",
                               commodity_col="Commodity", plot_type="pie",
                               ncols=2, height_per_row=400,
                               output_folder=None, filename=None):
    """
    Create interactive Plotly subplots for contribution analysis by metal.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns including Commodity, Activity, and Share_%.
    value_col : str, default 'Share_%'
        Column used for size (share or impact).
    label_col : str, default 'Activity'
        Column used for labels.
    commodity_col : str, default 'Commodity'
        Column indicating the metal or commodity.
    plot_type : {'pie', 'bar'}, default 'pie'
        Type of chart to show for each metal.
    ncols : int, default 2
        Number of columns of subplots.
    height_per_row : int, default 400
        Height per subplot row.
    output_folder : str, optional
        Folder to save the figure. Created if it doesn’t exist.
    filename : str, optional
        File name for saving (e.g., "contribution.html").
    """
    commodities = df[commodity_col].unique()
    nrows = (len(commodities) + ncols - 1) // ncols

    specs = [[{"type": "domain"} if plot_type=="pie" else {"type": "xy"}
              for _ in range(ncols)] for _ in range(nrows)]
    fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=commodities, specs=specs)

    # Add one trace per metal
    for i, commodity in enumerate(commodities):
        r, c = divmod(i, ncols)
        subset = df[df[commodity_col] == commodity].sort_values(by=value_col, ascending=False)

        if plot_type == "pie":
            fig.add_trace(
                go.Pie(
                    labels=subset[label_col],
                    values=subset[value_col],
                    hole=0.4,  # donut style
                    hovertext=subset["Product"] + "<br>" +
                              "Location: " + subset["Location"] + "<br>" +
                              "Impact: " + subset["Impact_score"].astype(str),
                    hoverinfo="label+percent+text",
                    textinfo="none"),
                row=r+1, col=c+1)
        else:
            fig.add_trace(
                go.Bar(
                    x=subset[label_col],
                    y=subset[value_col],
                    text=subset["Product"],
                    hovertext=("Location: " + subset["Location"] +
                               "<br>Impact: " + subset["Impact_score"].astype(str)),
                    hoverinfo="text+x+y"),
                row=r+1, col=c+1)

    fig.update_layout(
        title_text="First-tier Contribution Analysis by Metal",
        showlegend=False,
        height=nrows * height_per_row,
        margin=dict(t=80, b=40)
    )

    # --- Optional saving ---
    if output_folder and filename:
        os.makedirs(output_folder, exist_ok=True)
        save_path = os.path.join(output_folder, filename)
        fig.write_html(save_path)
        print(f"✅ Figure saved to: {save_path}")

    fig.show()


def plot_styled_boxplots_ore(df, agg_mapping, title, output_html, color_palette=None):
    """
    Create fully styled interactive boxplots grouped by aggregated midpoint categories.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'Commodity' column and multiple midpoint indicators.
    agg_mapping : dict
        Mapping of detailed midpoint categories to aggregated ones.
    title : str
        Title for the plot.
    output_html : str
        File path to export interactive HTML.
    color_palette : list
        Optional custom list of RGBA color strings.
    """

    # 1️⃣ Extract unit from any column that contains parentheses
    unit_pattern = re.compile(r"\((.*?)\)")
    all_units = [unit_pattern.findall(col) for col in df.columns if '(' in col and ')' in col]
    units_flat = [u[0] for u in all_units if u]
    unit = units_flat[0] if units_flat else "Impact units"

    # 2️⃣ Reshape data
    df_long = df.melt(id_vars="Commodity", var_name="Category", value_name="Value")

    # 3️⃣ Remove unit & trailing spaces for matching
    df_long["Category_clean"] = df_long["Category"].str.replace(r"\(.*\)", "", regex=True).str.strip()

    # 4️⃣ Map to aggregated category
    df_long["Aggregated"] = df_long["Category_clean"].replace(agg_mapping)

    # Remove any unmapped rows
    df_long = df_long[df_long["Aggregated"].notna() & df_long["Value"].notna()]

    # 5️⃣ Prepare categories
    categories = sorted(df_long["Aggregated"].unique())

    if color_palette is None:
        color_palette = [
            "rgba(93, 164, 214, 0.5)", "rgba(255, 144, 14, 0.5)",
            "rgba(44, 160, 101, 0.5)", "rgba(255, 65, 54, 0.5)",
            "rgba(207, 114, 255, 0.5)", "rgba(127, 96, 0, 0.5)",
            "rgba(100, 100, 255, 0.5)", "rgba(255, 200, 100, 0.5)"
        ] * 3

    # 6️⃣ Create boxplots
    fig = go.Figure()
    for cat, color in zip(categories, color_palette):
        subset = df_long[df_long["Aggregated"] == cat]
        fig.add_trace(go.Box(
            y=subset["Value"],
            name=cat,
            boxpoints="all",
            jitter=0.4,
            whiskerwidth=0.3,
            fillcolor=color,
            marker=dict(size=4, opacity=0.7),
            line=dict(width=1),
            hovertext=subset["Commodity"],  # facility name on hover
            hoverinfo="text+y"
        ))

    # 7️⃣ Layout and export
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=18)),
        yaxis=dict(
            title=unit,
            showgrid=True,
            zeroline=True,
            gridcolor="rgba(230,230,230,1)",
            zerolinecolor="rgba(180,180,180,1)"
        ),
        #xaxis=dict(title="Aggregated category"),
        paper_bgcolor="rgb(245,245,245)",
        plot_bgcolor="rgb(245,245,245)",
        margin=dict(l=50, r=50, t=80, b=100),
        showlegend=False,
        font=dict(family="Arial", size=12)
    )

    fig.write_html(output_html)
    print(f"✅ Plot saved to {output_html}")
    fig.show()


def plot_grade_relationship(
        df,
        x_col="grade_percent",
        y_col="total_energy_normalized",
        product_col="Main_product",
        label_col="facility_name",
        symbol_col="mining_processing_type",
        x_abs = 'grade (g/t)',
        y_abs = 'MJ/t',
        output_html="energy_vs_grade_plot.html"
    ):

    df_plot = df[(df[x_col] > 0) & (df[y_col] > 0)].copy()

    # Fit power law
    log_x = np.log10(df_plot[x_col])
    log_y = np.log10(df_plot[y_col])
    slope, intercept, r_value, _, _ = linregress(log_x, log_y)
    x_sorted = np.sort(df_plot[x_col])
    y_fit = 10**intercept * x_sorted**slope

    # ---- Symbol mapping (BLACK symbols only for mining) ----
    symbol_categories = df_plot[symbol_col].astype("category")
    df_plot["_symbol_code"] = symbol_categories.cat.codes
    symbol_map = dict(enumerate(symbol_categories.cat.categories))

    # ---- Colors for Main_product ----
    pastel_palette = px.colors.qualitative.Pastel
    products = df_plot[product_col].unique()
    color_cycle = itertools.cycle(pastel_palette)
    product_to_color = {p: next(color_cycle) for p in products}

    fig = go.Figure()

    # ---------------------------------------------------------
    # 1) MINING TYPE LEGEND  (symbols ONLY, black)
    # ---------------------------------------------------------
    for code, name in symbol_map.items():
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode="markers",
            marker=dict(symbol=code, size=12, color="black"),
            name=f"{name}",
            legendgroup="mining",
            showlegend=True
        ))

    # ---------------------------------------------------------
    # 2) MAIN PRODUCT LEGEND (colors ONLY, one fixed symbol)
    # ---------------------------------------------------------
    for product in products:
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode="markers",
            marker=dict(
                symbol="circle",       # <-- SYMBOL FIXE POUR LES COULEURS
                size=12,
                color=product_to_color[product],
                line=dict(color="black", width=1)
            ),
            name=f"{product}",
            legendgroup="product",
            showlegend=True
        ))

    # ---------------------------------------------------------
    # 3) REAL DATA POINTS  (symbol = mining, color = product)
    # ---------------------------------------------------------
    for _, row in df_plot.iterrows():
        fig.add_trace(go.Scatter(
            x=[row[x_col]],
            y=[row[y_col]],
            mode="markers",
            name=row[product_col],
            legendgroup="data",
            showlegend=False,          # <-- IMPORTANT : PAS DANS LA LÉGENDE
            marker=dict(
                symbol=row["_symbol_code"],
                color=product_to_color[row[product_col]],
                size=11,
                line=dict(color="black", width=1)
            ),
            text=row[label_col],
            hovertemplate=(
                f"Facility: {row[label_col]}<br>"
                f"{x_col}: {row[x_col]}<br>"
                f"{y_col}: {row[y_col]}"
            ),
        ))

    # ---------------------------------------------------------
    # 4) FIT CURVE (legend separate)
    # ---------------------------------------------------------
    fig.add_trace(go.Scatter(
        x=x_sorted,
        y=y_fit,
        mode="lines",
        name="Fit: y = {:.2e} × x^{:.2f}<br>R² = {:.2f}".format(
            10**intercept, slope, r_value**2
        ),
        line=dict(color="black", width=2),
        legendgroup="fit",
        showlegend=True
    ))

    # ---------------------------------------------------------
    # Layout (white background)
    # ---------------------------------------------------------
    fig.update_layout(
        title="",
        xaxis_title=x_abs,
        yaxis_title=y_abs,
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="black"),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            groupclick="toggleitem"
        ),
        xaxis=dict(gridcolor="lightgray"),
        yaxis=dict(
            gridcolor="lightgray",
            rangemode="tozero",
            tickformat='d'
        )
    )

    fig.write_html(output_html)
    fig.show()


def plot_energy_mix_archetypes(df, nrj_subflow, savepath=None,
                                          figsize=(14, 10), dpi=300):
    """
    Compute statistically correct energy mix archetypes and generate
    publication-ready 2x2 stacked bar plots.

    Method:
    --------
    1. Map subflow_type -> energy_group
    2. Aggregate raw MJ per site per energy_group
    3. Compute MEAN MJ across sites for each mining_processing_type & main_product
    4. Normalize MEAN MJ so bars always sum to 1
    5. Plot 2x2 stacked bars with n = number of sites
    6. Export to PNG / SVG / PDF if savepath provided

    Parameters
    ----------
    df : DataFrame
        Must include columns:
        ['site_id','mining_processing_type','Main_product','subflow_type','value_MJ']

    nrj_subflow : dict
        Mapping from raw subflow_type to grouped category.

    savepath : str or None
        If provided, exports PNG, SVG and PDF at this location (without extension).

    figsize : tuple
        Figure size.

    dpi : int
        Resolution for export.

    Returns
    -------
    stats_df : DataFrame
        mean, min, max, std for each archetype and energy group
    """

    df = df.copy()

    # ----------------------------------------------------------
    # 1) Map the subflow types
    # ----------------------------------------------------------
    df['energy_group'] = df['subflow_type'].map(nrj_subflow).fillna("Other")

    # ----------------------------------------------------------
    # 2) Aggregate at site level
    # ----------------------------------------------------------
    site_energy = (
        df.groupby(['site_id', 'mining_processing_type', 'Main_product',
                    'energy_group'])['value_MJ']
        .sum()
        .reset_index()
    )

    # Count sites per archetype
    site_counts = (
        site_energy.groupby(['mining_processing_type', 'Main_product'])['site_id']
        .nunique()
        .reset_index(name='n_sites')
    )

    # ----------------------------------------------------------
    # 3) Compute MEAN MJ across sites for each archetype (not shares yet)
    # ----------------------------------------------------------
    mean_energy = (
        site_energy.groupby(['mining_processing_type', 'Main_product',
                             'energy_group'])['value_MJ']
        .mean()
        .reset_index(name='mean_MJ')
    )

    # ----------------------------------------------------------
    # 4) Normalize the mean MJ so bars sum to 1
    # ----------------------------------------------------------
    mean_energy['share'] = (
        mean_energy.groupby(['mining_processing_type', 'Main_product'])['mean_MJ']
        .transform(lambda x: x / x.sum())
    )

    # Pivot for plotting
    pivot = mean_energy.pivot_table(
        index=['mining_processing_type', 'Main_product'],
        columns='energy_group',
        values='share',
        fill_value=0
    )

    mining_types = pivot.index.get_level_values(0).unique()
    energy_groups = pivot.columns

    # Colors
    colors = plt.get_cmap("tab20").colors[:len(energy_groups)]

    # ----------------------------------------------------------
    # 5) Plot
    # ----------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=figsize, sharey=True)
    axes = axes.flatten()

    for ax, mtype in zip(axes, mining_types):

        sub = pivot.loc[mtype]

        sub.plot(kind='bar', stacked=True, ax=ax,
                 color=colors, width=0.85, edgecolor="black")

        ax.set_title(mtype, fontsize=15)
        ax.tick_params(axis='x', labelsize=14)
        ax.set_ylabel("Percentage")
        ax.set_xlabel('')
        ax.margins(y=0.2)

        # Remove subplot legend
        ax.legend().set_visible(False)

        # Add n = X labels
        for idx, product in enumerate(sub.index):
            n_sites = site_counts.query(
                "mining_processing_type == @mtype and Main_product == @product"
            )['n_sites'].values[0]

            ax.text(idx, 1.05, f"n={n_sites}", ha='center', fontsize=11)

    # Hide additional empty axes (if <4 mining types)
    for ax in axes[len(mining_types):]:
        ax.set_visible(False)

    # One shared legend
    fig.legend(energy_groups,loc='lower center', bbox_to_anchor=(0.5, -0.1), ncol=4, fontsize=14)
    plt.tight_layout()

    # ----------------------------------------------------------
    # 6) Save figure if requested
    # ----------------------------------------------------------
    if savepath is not None:
        base = os.path.splitext(savepath)[0]

        fig.savefig(base + ".png", dpi=dpi, bbox_inches="tight")
        fig.savefig(base + ".svg", dpi=dpi, bbox_inches="tight")
        #fig.savefig(base + ".pdf", dpi=dpi, bbox_inches="tight")

        #print(f"Files saved:\n{base}.png\n{base}.svg\n{base}.pdf")

    plt.show()

    # ----------------------------------------------------------
    # 7) Return detailed statistics
    # ----------------------------------------------------------
    stats_df = (
        site_energy.groupby(['mining_processing_type','Main_product','energy_group'])['value_MJ']
        .agg(['mean','min','max','std'])
        .reset_index()
    )

    return stats_df