import plotly.graph_objects as go
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


def plot_lcia_boxes_for_reference_product_not_colored(
    df,
    reference_product,
    title=None,
    yaxis_label=None,
    output_html=None,
    output_image=None,  # ← NEW
    image_format="pdf",  # ← NEW: "pdf", "svg", "png"
    image_scale=3,  # ← NEW: for PNG
    others_threshold=0.05,
    benchmark_offset=0.3,
):
    """
    LCIA comparison plot:
    - MetalliCan: boxplots (site variability)
    - Ecoinvent / Regioinvent: median benchmark points (shifted left)
    - Impact categories sorted by importance
    - Small contributors grouped as 'Others'
    """

    ID_COLS = ["Source", "Reference product", "Commodity"]
    impact_cols = [c for c in df.columns if c not in ID_COLS]

    # --------------------------------------------------
    # Filter reference product
    # --------------------------------------------------
    df_rp = df[df["Reference product"] == reference_product].copy()
    if df_rp.empty:
        raise ValueError(f"No data for Reference product = {reference_product}")

    # --------------------------------------------------
    # Long format
    # --------------------------------------------------
    df_long = df_rp.melt(
        id_vars=ID_COLS,
        value_vars=impact_cols,
        var_name="Category",
        value_name="Value",
    ).dropna(subset=["Value"])

    # --------------------------------------------------
    # Group small contributors into "Others"
    # (based on MetalliCan only)
    # --------------------------------------------------
    mc_totals = (
        df_long[df_long["Source"] == "MetalliCan"]
        .groupby("Category")["Value"]
        .sum()
    )

    total_impact = mc_totals.sum()
    small_cats = mc_totals[mc_totals / total_impact < others_threshold].index.tolist()

    df_long["Category_plot"] = df_long["Category"]
    df_long.loc[df_long["Category"].isin(small_cats), "Category_plot"] = "Others"

    df_long = (
        df_long
        .groupby(ID_COLS + ["Category_plot"], as_index=False)["Value"]
        .sum()
    )

    # --------------------------------------------------
    # Sort categories by importance
    # --------------------------------------------------
    order_df = (
        df_long[df_long["Source"] == "MetalliCan"]
        .groupby("Category_plot", as_index=False)["Value"]
        .sum()
        .sort_values("Value", ascending=False)
    )

    category_order = order_df["Category_plot"].tolist()
    cat_to_x = {cat: i for i, cat in enumerate(category_order)}
    df_long["x_pos"] = df_long["Category_plot"].map(cat_to_x)

    # --------------------------------------------------
    # Benchmark medians
    # --------------------------------------------------
    bench = (
        df_long[df_long["Source"].isin(["Ecoinvent", "Regioinvent"])]
        .groupby(["Source", "Category_plot"], as_index=False)["Value"]
        .median()
    )

    fig = go.Figure()

    # --------------------------------------------------
    # MetalliCan boxplots
    # --------------------------------------------------
    mc = df_long[df_long["Source"] == "MetalliCan"]
    fig.add_trace(
        go.Box(
            x=mc["x_pos"],
            y=mc["Value"],
            boxpoints="all",
            jitter=0.35,
            fillcolor="rgba(31,119,180,0.45)",
            marker=dict(size=4, opacity=0.6),
            line=dict(color="black", width=1),
            hovertext=mc["Commodity"],
            hoverinfo="text+y",
            name="MetalliCan",
        )
    )

    # --------------------------------------------------
    # Benchmarks (shifted left)
    # --------------------------------------------------
    for src, symbol, color, size in [
        ("Ecoinvent", "diamond", "#d8b365", 10),
        ("Regioinvent", "star", "#5ab4ac", 10),
    ]:
        sub = bench[bench["Source"] == src].copy()
        if sub.empty:
            continue

        sub["x_pos"] = sub["Category_plot"].map(cat_to_x) - benchmark_offset

        fig.add_trace(
            go.Scatter(
                x=sub["x_pos"],
                y=sub["Value"],
                mode="markers",
                marker=dict(
                    symbol=symbol,
                    size=size,
                    color=color,
                    line=dict(width=1.6, color="black"),
                ),
                name=f"{src}",
            )
        )


    # --------------------------------------------------
    # Layout — CLEAN, NO DUPLICATES
    # --------------------------------------------------

    fig.update_layout(
        xaxis=dict(
            tickmode="array",
            tickvals=list(cat_to_x.values()),
            ticktext=list(cat_to_x.keys()),
            tickangle=-35,
            tickfont=dict(size=16, color="black"),
            ticklabelposition="outside",
            automargin=True,
        ),

        yaxis=dict(
            title=dict(
                text=yaxis_label or "Impact value",
                font=dict(color="black", size=20),
            ),
            tickfont=dict(color="black"),
            showgrid=False,     # ← removes horizontal lines
        ),

        plot_bgcolor="white",
        paper_bgcolor="white",

        margin=dict(l=80, r=40, t=80, b=180),

        legend=dict(
            title=dict(
                text="",
                font=dict(color="black", size=24),
            ),
            font=dict(color="black", size=18),  # ← bigger legend text
            orientation="v",                    # ← vertical legend
            yanchor="top",
            y=0.98,
            xanchor="right",
            x=0.98,
            bgcolor="rgba(255,255,255,0.9)",    # optional but very readable
            bordercolor="black",
            borderwidth=0.8,
        )

        )

    # --------------------------------------------------
    # Export
    # --------------------------------------------------
    if output_html:
        fig.write_html(output_html)

    if output_image:
        fig.write_image(
            f"{output_image}.{image_format}",
            scale=image_scale if image_format == "png" else 1
        )

    fig.show()


def plot_lcia_boxes_for_reference_product(
    df,
    reference_product,
    title=None,
    yaxis_label=None,
    output_html=None,
    output_image=None,  # ← NEW
    image_format="pdf",  # ← NEW: "pdf", "svg", "png"
    image_scale=3,  # ← NEW: for PNG
    others_threshold=0.05,
    benchmark_offset=0.3,
):
    """
    LCIA comparison plot:
    - MetalliCan: boxplots (site variability)
    - Ecoinvent / Regioinvent: median benchmark points (shifted left)
    - Impact categories sorted by importance
    - Small contributors grouped as 'Others'
    """

    ID_COLS = ["Source", "Reference product", "Commodity"]
    impact_cols = [c for c in df.columns if c not in ID_COLS]

    # --------------------------------------------------
    # Filter reference product
    # --------------------------------------------------
    df_rp = df[df["Reference product"] == reference_product].copy()
    if df_rp.empty:
        raise ValueError(f"No data for Reference product = {reference_product}")

    # --------------------------------------------------
    # Long format
    # --------------------------------------------------
    df_long = df_rp.melt(
        id_vars=ID_COLS,
        value_vars=impact_cols,
        var_name="Category",
        value_name="Value",
    ).dropna(subset=["Value"])

    # --------------------------------------------------
    # Group small contributors into "Others"
    # (based on MetalliCan only)
    # --------------------------------------------------
    mc_totals = (
        df_long[df_long["Source"] == "MetalliCan"]
        .groupby("Category")["Value"]
        .sum()
    )

    total_impact = mc_totals.sum()
    small_cats = mc_totals[mc_totals / total_impact < others_threshold].index.tolist()

    df_long["Category_plot"] = df_long["Category"]
    df_long.loc[df_long["Category"].isin(small_cats), "Category_plot"] = "Others"

    df_long = (
        df_long
        .groupby(ID_COLS + ["Category_plot"], as_index=False)["Value"]
        .sum()
    )

    # --------------------------------------------------
    # Sort categories by importance
    # --------------------------------------------------
    order_df = (
        df_long[df_long["Source"] == "MetalliCan"]
        .groupby("Category_plot", as_index=False)["Value"]
        .sum()
        .sort_values("Value", ascending=False)
    )

    category_order = order_df["Category_plot"].tolist()
    cat_to_x = {cat: i for i, cat in enumerate(category_order)}
    df_long["x_pos"] = df_long["Category_plot"].map(cat_to_x)

    # --------------------------------------------------
    # Benchmark medians
    # --------------------------------------------------
    bench = (
        df_long[df_long["Source"].isin(["Ecoinvent", "Regioinvent"])]
        .groupby(["Source", "Category_plot"], as_index=False)["Value"]
        .median()
    )

    fig = go.Figure()

    # --------------------------------------------------
    # MetalliCan boxplots
    # --------------------------------------------------
    impact_colors = {
    "Climate change": "#d73027",
    "Land transformation": "#1a9850",
    "Marine acidification": "#4575b4",
    "Terrestrial acidification": "#abd9e9",
    "Photochemical ozone": "#fee090",
    "Freshwater ecotoxicity": "#f1b6da",
    "Particulate matter formation": "#8073ac",
    "Human toxicity non-cancer": "#e08214",
    "Others": "#999999"
}


    mc = df_long[df_long["Source"] == "MetalliCan"]

    for cat, color in impact_colors.items():
        mc_cat = mc[mc["Category_plot"] == cat]
        if mc_cat.empty:
            continue

        fig.add_trace(
            go.Box(
                x=mc_cat["x_pos"],
                y=mc_cat["Value"],
                boxpoints="all",
                jitter=0.35,

                # 🔹 BOX styling (category color)
                fillcolor=color,
                line=dict(color="black", width=1),

                # 🔹 POINT styling (neutral)
                marker=dict(
                    color="rgba(80,80,80,0.65)",   # grey points
                    size=5,
                ),

                hovertext=mc_cat["Commodity"],
                hoverinfo="text+y",

                showlegend=False,   # ← VERY IMPORTANT
            )
        )

    # MetalliCan boxes (no color semantics)
    fig.add_trace(
        go.Scatter(
            x=[None], y=[None],
            mode="markers",
            marker=dict(
                symbol="square",
                size=16,
                color="rgba(0,0,0,0)",     # transparent
                line=dict(color="black", width=1.5),
            ),
            name="MetalliCan range",
            showlegend=True,
        )
    )

    # MetalliCan sites (grey points)
    fig.add_trace(
        go.Scatter(
            x=[None], y=[None],
            mode="markers",
            marker=dict(
                symbol="circle",
                size=10,
                color="rgba(80,80,80,0.7)",
            ),
            name="MetalliCan sites",
            showlegend=True,
        )
    )

    # --------------------------------------------------
    # Benchmarks (shifted left)
    # --------------------------------------------------
    for src, symbol, color, size in [
        ("Ecoinvent", "diamond", "#d8b365", 10),
        ("Regioinvent", "star", "#5ab4ac", 10),
    ]:
        sub = bench[bench["Source"] == src].copy()
        if sub.empty:
            continue

        sub["x_pos"] = sub["Category_plot"].map(cat_to_x) - benchmark_offset

        fig.add_trace(
            go.Scatter(
                x=sub["x_pos"],
                y=sub["Value"],
                mode="markers",
                marker=dict(
                    symbol=symbol,
                    size=size,
                    color=color,
                    line=dict(width=1.6, color="black"),
                ),
                name=f"{src}",
            )
        )


    # --------------------------------------------------
    # Layout — CLEAN, NO DUPLICATES
    # --------------------------------------------------

    fig.update_layout(
        xaxis=dict(
            tickmode="array",
            tickvals=list(cat_to_x.values()),
            ticktext=list(cat_to_x.keys()),
            tickangle=-35,
            tickfont=dict(size=16, color="black"),
            ticklabelposition="outside",
            automargin=True,
        ),

        yaxis=dict(
            title=dict(
                text=yaxis_label or "Impact value",
                font=dict(color="black", size=20),
            ),
            tickfont=dict(color="black"),
            showgrid=False,     # ← removes horizontal lines
        ),

        plot_bgcolor="white",
        paper_bgcolor="white",

        margin=dict(l=80, r=40, t=80, b=180),

        legend=dict(
            title=dict(
                text="",
                font=dict(color="black", size=24),
            ),
            font=dict(color="black", size=18),  # ← bigger legend text
            orientation="v",                    # ← vertical legend
            yanchor="top",
            y=0.98,
            xanchor="right",
            x=0.98,
            bgcolor="rgba(255,255,255,0.9)",    # optional but very readable
            bordercolor="black",
            borderwidth=0.8,
        )

        )

    # --------------------------------------------------
    # Export
    # --------------------------------------------------
    if output_html:
        fig.write_html(output_html)

    if output_image:
        fig.write_image(
            f"{output_image}.{image_format}",
            scale=image_scale if image_format == "png" else 1
        )

    fig.show()


def plot_stacked_lcia_by_site(
    bgf_tech,
    bgf_bio,
    agf_tech,
    agf_bio,
    impact_category,
    reference_product,
    production_col,
    title=None,
    figsize=(10, 6),
    height_min=0.2,
    height_max=1.6,
    xlabel=None,
    output_path=None,
    dpi=600,
):
    """
    Horizontal stacked LCIA barplot with variable bar height (Matplotlib).
    Linear min–max normalization for the y axis

    - Direct vs inferred distinguished by color + hatching
    - Bar height proportional to production volume
    - Exportable to publication-ready formats (PNG, PDF, SVG)

    Parameters
    ----------
    output_path : str, optional
        Path to save figure (extension defines format)
    dpi : int
        Resolution for raster formats (PNG)
    """

    # --------------------------------------------------
    # Filter reference product
    # --------------------------------------------------
    def _filt(df):
        return df[df["Reference product"] == reference_product].copy()

    bgf_tech = _filt(bgf_tech)
    bgf_bio  = _filt(bgf_bio)
    agf_tech = _filt(agf_tech)
    agf_bio  = _filt(agf_bio)

    # --------------------------------------------------
    # Merge datasets
    # --------------------------------------------------
    df = (
        bgf_tech[["Commodity", impact_category, production_col]]
        .rename(columns={impact_category: "tech_direct"})
        .merge(
            agf_tech[["Commodity", impact_category]]
            .rename(columns={impact_category: "tech_total"}),
            on="Commodity",
            how="inner",
        )
        .merge(
            bgf_bio[["Commodity", impact_category]]
            .rename(columns={impact_category: "bio_direct"}),
            on="Commodity",
            how="inner",
        )
        .merge(
            agf_bio[["Commodity", impact_category]]
            .rename(columns={impact_category: "bio_total"}),
            on="Commodity",
            how="inner",
        )
    )

    # --------------------------------------------------
    # Compute inferred contributions
    # --------------------------------------------------
    df["tech_inferred"] = df["tech_total"] - df["tech_direct"]
    df["bio_inferred"]  = df["bio_total"]  - df["bio_direct"]

    for c in ["tech_inferred", "bio_inferred"]:
        df.loc[df[c] < 0, c] = 0.0

    # --------------------------------------------------
    # Sort by production volume
    # --------------------------------------------------
    df = df.sort_values(production_col, ascending=False)
    bar_spacing=0.15

    sites = df["Commodity"].values
    #y_pos = np.arange(len(df))



    # --------------------------------------------------
    # Production → bar height mapping
    # --------------------------------------------------
    prod = df[production_col].astype(float).values

    if prod.max() == prod.min():
        heights = np.full_like(prod, (height_min + height_max) / 2)
    else:
        heights = height_min + (height_max - height_min) * (
            (prod - prod.min()) / (prod.max() - prod.min())
        )

    y_pos = np.zeros(len(df))
    for i in range(1, len(df)):
        y_pos[i] = y_pos[i-1] + heights[i-1] + bar_spacing

    # --------------------------------------------------
    # Plot
    # --------------------------------------------------
    fig, ax = plt.subplots(figsize=figsize)

    left = np.zeros(len(df))

    def _stack(values, label, color, hatch=None):
        nonlocal left
        ax.barh(
            y_pos,
            values,
            left=left,
            height=heights,
            label=label,
            color=color,
            hatch=hatch,
            edgecolor="black",
            linewidth=0.5,
        )
        left += values

    _stack(df["tech_direct"],   "Indirect impacts",   "#4C78A8")
    _stack(df["tech_inferred"], "Inferred indirect impacts", "#9ECAE9", hatch="///")
    _stack(df["bio_direct"],    "Direct impacts",      "#54A24B")
    _stack(df["bio_inferred"],  "Inferred direct impacts",    "#A1D99B", hatch="\\\\\\")

    # --------------------------------------------------
    # Axes & layout
    # --------------------------------------------------
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sites)
    ax.invert_yaxis()
    ax.tick_params(axis="x", labelsize=12)
    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel("")

    #ax.set_title(
    #    title or f"{impact_category} – {reference_product}",
    #    fontsize=12
    #)

    #ax.grid(axis="x", linestyle="--", alpha=0.4)

    ax.legend(
        loc="upper right",
        fontsize=16
        #bbox_to_anchor=(0.5, -0.15),
        #ncol=,
        #frameon=False,
    )

    plt.tight_layout()

    # --------------------------------------------------
    # Export
    # --------------------------------------------------
    if output_path:
        plt.savefig(
            output_path,
            dpi=dpi,
            bbox_inches="tight",
            facecolor="white",
        )

    plt.show()


def plot_total_lcia_by_site_multi_commodities(
    df,
    impact_category,
    reference_products,
    title=None,
    figsize=(13, 9),
    ref_gap=2.5,
    log_x=True,
    xlabel=None,
    output_path=None,
    dpi=600,
    ref_colors=None,
):
    """
    Multi-commodity LCIA comparison (TOTAL score only).

    - Bars: MetalliCan site-level results
    - Vertical lines: Ecoinvent (solid) & Regioinvent (dashed)
    - Bar color: reference product (commodity)
    - Log x-axis (valid, no stacking)
    - Y-axis: site names
    """

    EPS = 1e-12
    fig, ax = plt.subplots(figsize=figsize)

    y_positions = []
    y_labels = []

    current_y = 0

    # --------------------------------------------------
    # Default color palette
    # --------------------------------------------------
    if ref_colors is None:
        base_colors = [
            "#4C78A8",  # blue
            "#54A24B",  # green
            "#F58518",  # orange
            "#B279A2",  # purple
            "#E45756",  # red
            "#72B7B2",  # teal
        ]
        ref_colors = {
            ref: base_colors[i % len(base_colors)]
            for i, ref in enumerate(reference_products)
        }

    # --------------------------------------------------
    # Log tick formatter
    # --------------------------------------------------
    def log_tick_formatter(x, pos):
        if x <= EPS:
            return "0"
        return f"$10^{{{int(np.log10(x))}}}$"

    # --------------------------------------------------
    # Loop over reference products
    # --------------------------------------------------
    for ref in reference_products:

        df_ref = df[df["Reference product"] == ref].copy()
        if df_ref.empty:
            continue

        color = ref_colors.get(ref, "#4C78A8")

        # -----------------------------
        # Benchmarks (mean per source)
        # -----------------------------
        benchmarks = (
            df_ref[df_ref["Source"].isin(["Ecoinvent", "Regioinvent"])]
            .groupby("Source")[impact_category]
            .mean()
        )

        # -----------------------------
        # MetalliCan site results
        # -----------------------------
        df_sites = df_ref[df_ref["Source"] == "MetalliCan"].copy()
        if df_sites.empty:
            continue

        df_sites = df_sites.sort_values(impact_category, ascending=False)

        y_start = current_y

        for _, row in df_sites.iterrows():
            val = max(row[impact_category], EPS)

            ax.barh(
                current_y,
                val,
                height=0.7,
                color=color,
                edgecolor="black",
                linewidth=0.5,
            )

            y_positions.append(current_y)
            y_labels.append(row["Commodity"])
            current_y += 1

        y_end = current_y - 1

        # -----------------------------
        # Plot benchmark vertical lines
        # -----------------------------
        if "Ecoinvent" in benchmarks:
            ax.vlines(
            benchmarks["Ecoinvent"],
            y_start - 0.4,
            y_end + 0.4,
            colors='black',
            linestyles="solid",
            linewidth=2,
            alpha=0.9,
        )


        if "Regioinvent" in benchmarks:
            ax.vlines(
            benchmarks["Regioinvent"],
            y_start - 0.4,
            y_end + 0.4,
            colors='black',
            linestyles="dashed",
            linewidth=2,
            alpha=0.9,
        )


        current_y += ref_gap

    # --------------------------------------------------
    # Axes & layout
    # --------------------------------------------------
    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels)
    ax.tick_params(axis="x", labelsize=12)
    ax.tick_params(axis="y", labelsize=9)
    ax.invert_yaxis()

    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel("")

    if log_x:
        ax.set_xscale("log")
        ax.set_xlim(left=EPS)
        ax.xaxis.set_major_formatter(FuncFormatter(log_tick_formatter))

    if title:
        ax.set_title(title, fontsize=15)

    ax.grid(axis="x", linestyle="--", alpha=0.4)

    # --------------------------------------------------
    # Legend 1: commodities (colors)
    # --------------------------------------------------
    # commodity_handles = [
    #     plt.Line2D([0], [0], color=ref_colors[ref], lw=8, label=ref)
    #     for ref in reference_products
    #     if ref in ref_colors
    # ]
    #
    # legend1 = ax.legend(
    #     handles=commodity_handles,
    #     fontsize=13,
    #     loc="upper center",
    #     bbox_to_anchor=(0.5, -0.12),
    #     ncol=min(3, len(commodity_handles)),
    #     frameon=False,
    # )
    #
    # ax.add_artist(legend1)

    # --------------------------------------------------
    # Legend 2: benchmark line styles
    # --------------------------------------------------
    benchmark_handles = [
        plt.Line2D([0], [0], color="black", lw=2, linestyle="solid", label="Ecoinvent"),
        plt.Line2D([0], [0], color="black", lw=2, linestyle="dashed", label="Regioinvent"),
    ]

    # --- commodity color legend ---
    commodity_handles = [
        Line2D(
            [0], [0],
            color=ref_colors[ref],
            lw=8,
            label=ref
        )
        for ref in reference_products
        if ref in ref_colors
    ]

    # --- benchmark line legend ---
    benchmark_handles = [
        Line2D([0], [0], color="black", lw=2, linestyle="solid", label="Ecoinvent"),
        Line2D([0], [0], color="black", lw=2, linestyle="dashed", label="Regioinvent"),
    ]

    # --- combined legend ---
    ax.legend(
        handles=commodity_handles + benchmark_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=3,
        frameon=False,
        fontsize=16,
    )


    plt.tight_layout()

    # --------------------------------------------------
    # Export
    # --------------------------------------------------
    if output_path:
        plt.savefig(
            output_path,
            dpi=dpi,
            bbox_inches="tight",
            facecolor="white",
        )

    plt.show()


def plot_stacked_contribution_by_activity(
    df,
    impact_col="share_%",
    commodity_col="reference_product_market",
    name_col="name",
    flow_type_col="flow_type",
    commodity_order=None,
    act_to_hex=None,                     # dict: {activity_name: "#RRGGBB"}
    act_to_plot=None,                    # list of activities to display (order preserved)
    top_n=12,                            # used only if act_to_plot=None
    others_label="Others",
    hatch_technosphere="///",
    hatch_biosphere=None,
    bar_spacing=0.15,
    figsize=(11, 7),
    xlabel=None,
    title=None,
    normalize_to_percent=False,          # if True: each commodity is normalized to 100%
    legend=True,
    legend_fontsize=9,
    legend_ncol=1,
    output_path=None,
    dpi=600,
):
    """
    Horizontal stacked contribution barplot across ALL commodities.

    - Colors encode activity (name)
    - Hatch encodes flow_type (Technosphere hatched; Biosphere solid)
    - Unselected activities grouped into 'Others'
    - Legend style: "Activity — Bio" and "Activity — Tech"
      BUT 'Others' appears only once.

    Required columns in df:
      - commodity_col
      - name_col
      - flow_type_col (expects e.g. 'Technosphere (first-tier)' and 'Biosphere (direct)')
      - impact_col
    """

    df = df.copy()

    # --- Basic checks
    for c in [commodity_col, name_col, flow_type_col, impact_col]:
        if c not in df.columns:
            raise ValueError(f"Missing column: '{c}'")

    # Coerce impacts to numeric
    df[impact_col] = pd.to_numeric(df[impact_col], errors="coerce").fillna(0.0)

    # --- Decide which activities to plot
    if act_to_plot is None:
        tmp = (
            df.groupby(name_col, as_index=False)[impact_col].sum()
            .assign(_abs=lambda x: x[impact_col].abs())
            .sort_values("_abs", ascending=False)
        )
        act_to_plot = tmp[name_col].head(top_n).tolist()
    act_to_plot = list(act_to_plot)  # preserve given order

    # --- Colors
    act_to_hex = act_to_hex or {}

    def _get_color(act, i):
        if act in act_to_hex:
            return act_to_hex[act]
        cmap = plt.get_cmap("tab20")
        return cmap(i % 20)

    # --- Group non-selected -> Others
    df["_act"] = df[name_col].where(df[name_col].isin(act_to_plot), others_label)

    # --- Aggregate
    agg = (
        df.groupby([commodity_col, "_act", flow_type_col], as_index=False)[impact_col]
        .sum()
        .rename(columns={impact_col: "impact"})
    )

    # --- Commodity order
    if commodity_order is not None:
        missing = set(commodity_order) - set(agg[commodity_col].unique())
        if missing:
            raise ValueError(f"Commodities not found in data: {missing}")
    else:
        commodity_order = (
            agg.groupby(commodity_col, as_index=False)["impact"].sum()
            .assign(_abs=lambda x: x["impact"].abs())
            .sort_values("_abs", ascending=False)[commodity_col]
            .tolist()
        )

    # --- Fixed bar heights (you don’t use production scaling here)
    heights = np.full(len(commodity_order), 0.7, dtype=float)

    # y positions with spacing
    y_pos = np.zeros(len(commodity_order), dtype=float)
    for i in range(1, len(commodity_order)):
        y_pos[i] = y_pos[i - 1] + heights[i - 1] + bar_spacing

    # --- Flow labels (expected)
    flow_bio = "Biosphere (direct)"
    flow_tech = "Technosphere (first-tier)"

    flow_types_present = set(agg[flow_type_col].unique().tolist())
    flow_order = []
    if flow_bio in flow_types_present:
        flow_order.append(flow_bio)
    if flow_tech in flow_types_present:
        flow_order.append(flow_tech)
    # any other flow types appended
    flow_order += [ft for ft in sorted(flow_types_present) if ft not in set(flow_order)]

    # --- Activity order
    acts_present = set(agg["_act"].unique().tolist())
    act_order = [a for a in act_to_plot if a in acts_present]
    if others_label in acts_present:
        act_order.append(others_label)

    # --- Pivot
    pivot = (
        agg.pivot_table(
            index=[commodity_col],
            columns=["_act", flow_type_col],
            values="impact",
            aggfunc="sum",
            fill_value=0.0,
        )
        .reindex(index=commodity_order, fill_value=0.0)
    )

    # Normalize each commodity to 100% if requested
    if normalize_to_percent:
        row_sums = pivot.sum(axis=1).replace(0, np.nan)
        pivot = (pivot.div(row_sums, axis=0) * 100.0).fillna(0.0)
        if xlabel is None:
            xlabel = "Contribution (%)"
    else:
        if xlabel is None:
            xlabel = impact_col

    # --- Plot (constrained_layout prevents right truncation)
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    left = np.zeros(len(commodity_order), dtype=float)

    # For legend control: avoid duplicates, and keep Others only once
    labeled = set()

    for i_act, act in enumerate(act_order):
        color = _get_color(act, i_act)

        for ft in flow_order:
            try:
                vals = pivot[(act, ft)].values
            except KeyError:
                vals = np.zeros(len(commodity_order), dtype=float)

            if np.allclose(vals, 0):
                continue

            hatch = hatch_technosphere if ft == flow_tech else hatch_biosphere

            # ---- Legend label logic ----
            if act == others_label:
                # One single "Others" entry in legend
                label = others_label if others_label not in labeled else "_nolegend_"
                if label != "_nolegend_":
                    labeled.add(others_label)
            else:
                suffix = "Tech" if ft == flow_tech else "Bio"
                label_full = f"{act}"
                label = label_full if label_full not in labeled else "_nolegend_"
                if label != "_nolegend_":
                    labeled.add(label_full)

            ax.barh(
                y_pos,
                vals,
                left=left,
                height=heights,
                color=color,
                hatch=hatch,
                edgecolor="black",
                linewidth=0.5,
                label=label,
            )
            left += vals

    # Axes
    ax.set_yticks(y_pos)
    ax.set_yticklabels(commodity_order)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel, fontsize=12)
    ax.tick_params(axis="x", labelsize=11)
    ax.tick_params(axis="y", labelsize=11)

    if title is not None:
        ax.set_title(title, fontsize=12)

    if legend:
        ax.legend(
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            frameon=False,
            fontsize=legend_fontsize,
            ncol=legend_ncol,
        )

    # Save
    if output_path:
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")

    plt.show()
