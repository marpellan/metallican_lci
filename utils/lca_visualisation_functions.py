import plotly.graph_objects as go
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_lcia_boxes_for_reference_product(
    df,
    reference_product,
    title=None,
    yaxis_label=None,
    output_html=None,
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
        ("Ecoinvent", "circle-open", "#e8c547", 10),
        ("Regioinvent", "square-open", "#30323d", 10),
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
        # title=dict(
        #     text=title or f"LCIA comparison – {reference_product}",
        #     x=0.5,
        #     font=dict(color="black", size=14),
        # ),
        # font=dict(color="black", size=12),

        xaxis=dict(
            tickmode="array",
            tickvals=list(cat_to_x.values()),
            ticktext=list(cat_to_x.keys()),
            tickfont=dict(color="black", size=16),
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

        margin=dict(l=80, r=40, t=80, b=120),

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

    if output_html:
        fig.write_html(output_html)

    fig.show()


def plot_stacked_lcia_by_site_matplotlib(
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

    ax.set_xlabel(impact_category, fontsize=14)
    ax.set_ylabel("")

    #ax.set_title(
    #    title or f"{impact_category} – {reference_product}",
    #    fontsize=12
    #)

    #ax.grid(axis="x", linestyle="--", alpha=0.4)

    ax.legend(
        loc="upper right",
        fontsize=14
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