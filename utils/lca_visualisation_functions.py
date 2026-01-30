import plotly.graph_objects as go
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


# def plot_lcia_boxes_for_reference_product(
#     df,
#     reference_product,
#     title=None,
#     yaxis_label=None,
#     output_html=None,
#     output_image=None,
#     image_format="pdf",          # "pdf", "svg", "png"
#     image_scale=3,               # only for png
#     others_threshold=0.05,
#     benchmark_offset=0.3,
#     orientation="h",             # "v" (catégories sur x) or "h" (catégories sur y)
#     include_total=False,          # add "Total" = sum of categories
#     total_position="first",      # "first" or "last"
#     legend_mode="none",          # "full", "none", "only" (legend-only figure)
#     impact_colors=None,          # optionally override colors
#     show_points=True,            # show MetalliCan individual sites
# ):
#     """
#     LCIA comparison plot:
#     - MetalliCan: boxplots (site variability)
#     - Ecoinvent / Regioinvent: median benchmark points (shifted)
#     - Small contributors grouped as 'Others' (based on MetalliCan total share)
#     - Optional "Total" category (sum of all impact categories)
#     - Orientation vertical/horizontal
#     - Legend modes:
#         * "full": normal plot with legend
#         * "none": no legend (for subfigures repeated)
#         * "only": legend-only figure (to include once in LaTeX)
#     """
#
#     ID_COLS = ["Source", "Reference product", "Commodity"]
#     impact_cols = [c for c in df.columns if c not in ID_COLS]
#
#     df_rp = df[df["Reference product"] == reference_product].copy()
#     if df_rp.empty:
#         raise ValueError(f"No data for Reference product = {reference_product}")
#
#     # ---- long format
#     df_long = df_rp.melt(
#         id_vars=ID_COLS,
#         value_vars=impact_cols,
#         var_name="Category",
#         value_name="Value",
#     ).dropna(subset=["Value"])
#
#     # ---- add Total (before Others grouping)
#     if include_total:
#         totals = (
#             df_long.groupby(ID_COLS, as_index=False)["Value"]
#             .sum()
#             .assign(Category="Total")
#         )
#         df_long = pd.concat([df_long, totals], ignore_index=True)
#
#     # ---- group small contributors into Others (based on MetalliCan, excluding Total)
#     mc = df_long[(df_long["Source"] == "MetalliCan") & (df_long["Category"] != "Total")]
#     mc_totals = mc.groupby("Category")["Value"].sum()
#     total_impact = mc_totals.sum()
#
#     small_cats = []
#     if total_impact and total_impact != 0:
#         small_cats = mc_totals[(mc_totals / total_impact) < others_threshold].index.tolist()
#
#     df_long["Category_plot"] = df_long["Category"]
#     df_long.loc[df_long["Category"].isin(small_cats), "Category_plot"] = "Others"
#
#     # regroup (so Others becomes one category per site/source)
#     df_long = (
#         df_long.groupby(ID_COLS + ["Category_plot"], as_index=False)["Value"]
#         .sum()
#     )
#
#     # ---- order categories by importance (MetalliCan totals), keep Total position
#     order_base = (
#         df_long[(df_long["Source"] == "MetalliCan") & (df_long["Category_plot"] != "Total")]
#         .groupby("Category_plot", as_index=False)["Value"]
#         .sum()
#         .sort_values("Value", ascending=False)
#     )
#     category_order = order_base["Category_plot"].tolist()
#
#     if include_total and "Total" in df_long["Category_plot"].unique():
#         if total_position == "first":
#             category_order = ["Total"] + category_order
#         else:
#             category_order = category_order + ["Total"]
#
#     # ensure uniqueness (just in case)
#     seen = set()
#     category_order = [c for c in category_order if not (c in seen or seen.add(c))]
#
#     cat_to_pos = {cat: i for i, cat in enumerate(category_order)}
#     df_long["pos"] = df_long["Category_plot"].map(cat_to_pos)
#
#     # ---- benchmark medians
#     bench = (
#         df_long[df_long["Source"].isin(["Ecoinvent", "Regioinvent"])]
#         .groupby(["Source", "Category_plot"], as_index=False)["Value"]
#         .median()
#     )
#
#     # ---- default palette
#     if impact_colors is None:
#         impact_colors = {
#             "Total": "#000000",
#             "Climate change": "#d73027",
#             "Land transformation": "#1a9850",
#             "Marine acidification": "#4575b4",
#             "Terrestrial acidification": "#abd9e9",
#             "Photochemical ozone": "#fee090",
#             "Freshwater ecotoxicity": "#f1b6da",
#             "Particulate matter formation": "#8073ac",
#             "Human toxicity non-cancer": "#e08214",
#             "Others": "#999999",
#         }
#
#     # ---- legend-only output (single figure used in LaTeX once)
#     if legend_mode == "only":
#         fig = go.Figure()
#
#         fig.add_trace(
#             go.Scatter(
#                 x=[None], y=[None],
#                 mode="markers",
#                 marker=dict(
#                     symbol="square",
#                     size=16,
#                     color="rgba(0,0,0,0)",
#                     line=dict(color="black", width=1.5),
#                 ),
#                 name="MetalliCan range",
#                 showlegend=True,
#             )
#         )
#         fig.add_trace(
#             go.Scatter(
#                 x=[None], y=[None],
#                 mode="markers",
#                 marker=dict(
#                     symbol="circle",
#                     size=10,
#                     color="rgba(80,80,80,0.7)",
#                 ),
#                 name="MetalliCan sites",
#                 showlegend=True,
#             )
#         )
#         fig.add_trace(
#             go.Scatter(
#                 x=[None], y=[None],
#                 mode="markers",
#                 marker=dict(
#                     symbol="diamond",
#                     size=10,
#                     color="#d8b365",
#                     line=dict(width=1.6, color="black"),
#                 ),
#                 name="Ecoinvent",
#                 showlegend=True,
#             )
#         )
#         fig.add_trace(
#             go.Scatter(
#                 x=[None], y=[None],
#                 mode="markers",
#                 marker=dict(
#                     symbol="star",
#                     size=10,
#                     color="#5ab4ac",
#                     line=dict(width=1.6, color="black"),
#                 ),
#                 name="Regioinvent",
#                 showlegend=True,
#             )
#         )
#
#         fig.update_layout(
#             xaxis=dict(visible=False),
#             yaxis=dict(visible=False),
#             plot_bgcolor="white",
#             paper_bgcolor="white",
#             margin=dict(l=10, r=10, t=10, b=10),
#             legend=dict(
#                 orientation="h",
#                 yanchor="middle",
#                 y=0.5,
#                 xanchor="center",
#                 x=0.5,
#                 bgcolor="rgba(255,255,255,0)",
#                 borderwidth=0,
#                 font=dict(color="black", size=18),
#             ),
#         )
#
#         if output_image:
#             fig.write_image(f"{output_image}.{image_format}", scale=image_scale if image_format == "png" else 1)
#         if output_html:
#             fig.write_html(output_html)
#         fig.show()
#         return fig
#
#     # ---- normal figure
#     fig = go.Figure()
#
#     mc_only = df_long[df_long["Source"] == "MetalliCan"].copy()
#
#     # plot each category as one box trace (so we can color by category)
#     for cat in category_order:
#         mc_cat = mc_only[mc_only["Category_plot"] == cat]
#         if mc_cat.empty:
#             continue
#
#         # use category color if present, else fallback grey
#         color = impact_colors.get(cat, "#cccccc")
#
#         if orientation == "v":
#             x_vals = mc_cat["pos"]
#             y_vals = mc_cat["Value"]
#         else:
#             x_vals = mc_cat["Value"]
#             y_vals = mc_cat["pos"]
#
#         fig.add_trace(
#             go.Box(
#                 x=x_vals,
#                 y=y_vals,
#                 boxpoints="all" if show_points else False,
#                 jitter=0.35 if show_points else 0,
#                 fillcolor=color,
#                 line=dict(color="black", width=1),
#                 marker=dict(color="rgba(80,80,80,0.65)", size=5) if show_points else dict(color="rgba(0,0,0,0)"),
#                 hovertext=mc_cat["Commodity"],
#                 hoverinfo="text+x" if orientation == "h" else "text+y",
#                 showlegend=False,
#                 orientation="h" if orientation == "h" else "v",
#             )
#         )
#
#     # legend entries (dummy)
#     fig.add_trace(
#         go.Scatter(
#             x=[None], y=[None],
#             mode="markers",
#             marker=dict(symbol="square", size=16, color="rgba(0,0,0,0)", line=dict(color="black", width=1.5)),
#             name="MetalliCan range",
#             showlegend=(legend_mode == "full"),
#         )
#     )
#     fig.add_trace(
#         go.Scatter(
#             x=[None], y=[None],
#             mode="markers",
#             marker=dict(symbol="circle", size=10, color="rgba(80,80,80,0.7)"),
#             name="MetalliCan sites",
#             showlegend=(legend_mode == "full"),
#         )
#     )
#
#     # benchmarks shifted
#     for src, symbol, color, size in [
#         ("Ecoinvent", "diamond", "#d8b365", 10),
#         ("Regioinvent", "star", "#5ab4ac", 10),
#     ]:
#         sub = bench[bench["Source"] == src].copy()
#         if sub.empty:
#             continue
#
#         if orientation == "v":
#             sub["pos_shift"] = sub["Category_plot"].map(cat_to_pos) - benchmark_offset
#             fig.add_trace(
#                 go.Scatter(
#                     x=sub["pos_shift"],
#                     y=sub["Value"],
#                     mode="markers",
#                     marker=dict(symbol=symbol, size=size, color=color, line=dict(width=1.6, color="black")),
#                     name=src,
#                     showlegend=(legend_mode == "full"),
#                 )
#             )
#         else:
#             # in horizontal, shift along the categorical axis (y)
#             sub["pos_shift"] = sub["Category_plot"].map(cat_to_pos) - benchmark_offset
#             fig.add_trace(
#                 go.Scatter(
#                     x=sub["Value"],
#                     y=sub["pos_shift"],
#                     mode="markers",
#                     marker=dict(symbol=symbol, size=size, color=color, line=dict(width=1.6, color="black")),
#                     name=src,
#                     showlegend=(legend_mode == "full"),
#                 )
#             )
#
#     # axis ticks
#     tick_vals = list(cat_to_pos.values())
#     tick_text = list(cat_to_pos.keys())
#
#     if orientation == "v":
#         fig.update_layout(
#             xaxis=dict(
#                 tickmode="array",
#                 tickvals=tick_vals,
#                 ticktext=tick_text,
#                 tickangle=-35,
#                 tickfont=dict(size=16, color="black"),
#                 automargin=True,
#             ),
#             yaxis=dict(
#                 title=dict(text=yaxis_label or "Impact value", font=dict(color="black", size=20)),
#                 tickfont=dict(color="black"),
#                 showgrid=False,
#             ),
#         )
#     else:
#         fig.update_layout(
#         xaxis=dict(
#             title=dict(text=yaxis_label or "Impact value", font=dict(color="black", size=20)),
#             tickfont=dict(size=24, color="black"),
#             showgrid=False,
#
#             zeroline=False,          # ← CRUCIAL : supprime la ligne à x=0
#             showline=True,
#             linecolor="black",
#             linewidth=1.2,
#             mirror=True,             # ← ferme le cadre (haut + bas)
#         ),
#         yaxis=dict(
#             tickmode="array",
#             tickvals=tick_vals,
#             ticktext=tick_text,
#             tickfont=dict(size=24, color="black"),
#             automargin=True,
#             showgrid=False,
#
#             zeroline=False,          # ← symétrie, bonne pratique
#             showline=True,
#             linecolor="black",
#             linewidth=1.2,
#             mirror=True,             # ← ferme le cadre (gauche + droite)
#         ),
#     )
#
#     #PANEL_MARGIN = dict(l=160, r=40, t=40, b=60)  # try l=160–220 depending on your tick font
#
#     fig.update_layout(
#         title=dict(text=title or "", x=0.5),
#         plot_bgcolor="white",
#         paper_bgcolor="white",
#         margin=dict(l=80, r=40, t=60, b=160 if orientation == "v" else 80),
#         #margin=PANEL_MARGIN,
#         legend=dict(
#             orientation="h",   # if you keep legend in-plot, horizontal is usually cleaner
#             yanchor="top",
#             y=-0.25 if orientation == "v" else -0.20,
#             xanchor="center",
#             x=0.5,
#             bgcolor="rgba(255,255,255,0.9)",
#             bordercolor="black",
#             borderwidth=0.8,
#             font=dict(color="black", size=18),
#         ) if legend_mode == "full" else dict(),
#         showlegend=(legend_mode == "full"),
#     )
#
#     if output_html:
#         fig.write_html(output_html)
#
#     FIG_WIDTH = 900
#     FIG_HEIGHT = 520  # same for all EQ/HH category plots
#
#     if output_image:
#         # fig.write_image(
#         #     f"{output_image}.{image_format}",
#         #     scale=image_scale if image_format == "png" else 1
#         # )
#         fig.write_image(
#             f"{output_image}.{image_format}",
#             width=FIG_WIDTH,
#             height=FIG_HEIGHT,
#             scale=image_scale if image_format == "png" else 1,
#         )
#     #fig.show()
#     return fig
#
#
# def plot_lcia_boxes_for_total_damages(
#     df,
#     reference_products,                 # e.g. ["Doré", "Cu concentrate", "Ni concentrate"]
#     title="",
#     xaxis_label="Total impact",
#     output_html=None,
#     output_image=None,
#     image_format="pdf",
#     image_scale=3,
#     benchmark_offset=0.25,
#     show_points=True,
#     legend_mode="none",                 # "full" or "none"
#     order=None,                         # optional custom order for y
# ):
#     """
#     Plot total damages (sum of all impact categories) across multiple reference products.
#     - MetalliCan: boxplots across sites (one box per reference product)
#     - Ecoinvent/Regioinvent: median benchmark points (shifted slightly)
#     """
#
#     ID_COLS = ["Source", "Reference product", "Commodity"]
#     impact_cols = [c for c in df.columns if c not in ID_COLS]
#
#     d = df[df["Reference product"].isin(reference_products)].copy()
#     if d.empty:
#         raise ValueError("No data for selected reference_products.")
#
#     # Long -> compute Total per (Source, Reference product, Commodity)
#     d_long = d.melt(
#         id_vars=ID_COLS,
#         value_vars=impact_cols,
#         var_name="Category",
#         value_name="Value",
#     ).dropna(subset=["Value"])
#
#     d_total = (
#         d_long.groupby(["Source", "Reference product", "Commodity"], as_index=False)["Value"]
#         .sum()
#         .rename(columns={"Value": "Total"})
#     )
#
#     # Order on y
#     if order is None:
#         y_order = reference_products
#     else:
#         y_order = order
#
#     y_to_pos = {rp: i for i, rp in enumerate(y_order)}
#     d_total["y_pos"] = d_total["Reference product"].map(y_to_pos)
#
#     # Bench medians
#     bench = (
#         d_total[d_total["Source"].isin(["Ecoinvent", "Regioinvent"])]
#         .groupby(["Source", "Reference product"], as_index=False)["Total"]
#         .median()
#     )
#     bench["y_pos"] = bench["Reference product"].map(y_to_pos)
#
#     fig = go.Figure()
#
#     # --- MetalliCan boxplots (one per reference product)
#     mc = d_total[d_total["Source"] == "MetalliCan"].copy()
#
#     for rp in y_order:
#         sub = mc[mc["Reference product"] == rp]
#         if sub.empty:
#             continue
#
#         fig.add_trace(
#             go.Box(
#                 x=sub["Total"],
#                 y=[y_to_pos[rp]] * len(sub),
#                 orientation="h",
#                 boxpoints="all" if show_points else False,
#                 jitter=0.35 if show_points else 0,
#                 fillcolor="rgba(150,150,150,0.35)",
#                 line=dict(color="black", width=1),
#                 marker=dict(color="rgba(80,80,80,0.65)", size=5) if show_points else dict(color="rgba(0,0,0,0)"),
#                 hovertext=sub["Commodity"],
#                 hoverinfo="text+x",
#                 showlegend=False,
#             )
#         )
#
#     # dummy legend entries (optional)
#     fig.add_trace(
#         go.Scatter(
#             x=[None], y=[None],
#             mode="markers",
#             marker=dict(symbol="square", size=16, color="rgba(0,0,0,0)", line=dict(color="black", width=1.5)),
#             name="MetalliCan range",
#             showlegend=(legend_mode == "full"),
#         )
#     )
#     fig.add_trace(
#         go.Scatter(
#             x=[None], y=[None],
#             mode="markers",
#             marker=dict(symbol="circle", size=10, color="rgba(80,80,80,0.7)"),
#             name="MetalliCan sites",
#             showlegend=(legend_mode == "full"),
#         )
#     )
#
#     # --- Benchmarks (median points)
#     for src, symbol, color, size in [
#         ("Ecoinvent", "diamond", "#d8b365", 10),
#         ("Regioinvent", "star", "#5ab4ac", 10),
#     ]:
#         sub = bench[bench["Source"] == src].copy()
#         if sub.empty:
#             continue
#         sub["y_shift"] = sub["y_pos"] - benchmark_offset
#
#         fig.add_trace(
#             go.Scatter(
#                 x=sub["Total"],
#                 y=sub["y_shift"],
#                 mode="markers",
#                 marker=dict(symbol=symbol, size=size, color=color, line=dict(width=1.6, color="black")),
#                 name=src,
#                 showlegend=(legend_mode == "full"),
#             )
#         )
#
#     # Axes + frame (no zeroline, framed outside only)
#     tick_vals = list(y_to_pos.values())
#     tick_text = list(y_to_pos.keys())
#
#     #PANEL_MARGIN = dict(l=160, r=40, t=40, b=60)  # try l=160–220 depending on your tick font
#
#     fig.update_layout(
#         title=dict(text=title, x=0.5),
#         plot_bgcolor="white",
#         paper_bgcolor="white",
#         margin=dict(l=120, r=40, t=60, b=60),
#         #margin=PANEL_MARGIN,
#         showlegend=(legend_mode == "full"),
#
#         xaxis=dict(
#             type='log',
#             title=dict(text=xaxis_label, font=dict(color="black", size=20)),
#             tickfont=dict(size=24, color="black"),
#             showgrid=False,
#             zeroline=False,      # important
#             showline=True,
#             linecolor="black",
#             linewidth=1.2,
#             mirror=True,
#         ),
#         yaxis=dict(
#             tickmode="array",
#             tickvals=tick_vals,
#             ticktext=tick_text,
#             tickfont=dict(size=24, color="black"),
#             showgrid=False,
#             zeroline=False,
#             showline=True,
#             linecolor="black",
#             linewidth=1.2,
#             mirror=True,
#         ),
#     )
#
#     if output_html:
#         fig.write_html(output_html)
#
#     FIG_WIDTH = 900
#     FIG_HEIGHT = 520  # same for all EQ/HH category plots
#
#     if output_image:
#         # fig.write_image(
#         #     f"{output_image}.{image_format}",
#         #     scale=image_scale if image_format == "png" else 1
#         # )
#         fig.write_image(
#             f"{output_image}.{image_format}",
#             width=FIG_WIDTH,
#             height=FIG_HEIGHT,
#             scale=image_scale if image_format == "png" else 1,
#         )
#
#     return fig

import numpy as np
import pandas as pd
import plotly.graph_objects as go


# -----------------------------------------------------------------------------
# Helpers: enforce identical "plot area" (black frame) across ALL exported panels
# -----------------------------------------------------------------------------
def _format_log_ticktext(val: float) -> str:
    """Option A: 1, 10, 100, 1k, 10k, 100k, 1M ..."""
    if val >= 1e9:
        return f"{val/1e9:g}G"
    if val >= 1e6:
        return f"{val/1e6:g}M"
    if val >= 1e3:
        return f"{val/1e3:g}k"
    if val >= 1:
        return f"{val:g}"
    # For completeness, keep raw for <1 (rare in your use case)
    return f"{val:g}"


def _major_log_ticks_from_data(values, base=10):
    """
    Build major log ticks strictly at powers of 10 that cover the data range.
    Returns tickvals, ticktext.
    """
    vals = pd.to_numeric(pd.Series(values).dropna(), errors="coerce")
    vals = vals[vals > 0]
    if vals.empty:
        # fallback
        tickvals = [1, 10, 100, 1000, 10000]
        return tickvals, [_format_log_ticktext(v) for v in tickvals]

    vmin = float(vals.min())
    vmax = float(vals.max())

    pmin = int(np.floor(np.log(vmin) / np.log(base)))
    pmax = int(np.ceil(np.log(vmax) / np.log(base)))

    tickvals = [base ** p for p in range(pmin, pmax + 1)]
    ticktext = [_format_log_ticktext(v) for v in tickvals]
    return tickvals, ticktext


def apply_common_panel_layout(
    fig: go.Figure,
    *,
    width=900,
    height=520,
    # These margins control the OUTER whitespace. Fix them to make the INNER black frame identical.
    margin_l=140,
    margin_r=20,
    margin_t=50,
    margin_b=70,
    # Font sizes
    tick_font_size=22,
    axis_title_size=20,
    title_size=18,
    # Frame style
    frame_linewidth=1.2,
    frame_color="black",
    # Optionally force "no extra" margins from Plotly auto layout
    lock_automargins=True,
):
    """
    Make panels export with:
    - identical width/height
    - identical margins (=> identical plot area size, i.e., black frame)
    - consistent fonts and frame
    """
    # Layout
    fig.update_layout(
        width=width,
        height=height,
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=margin_l, r=margin_r, t=margin_t, b=margin_b),
        title=dict(x=0.5, font=dict(size=title_size, color="black")),
        showlegend=fig.layout.showlegend,
    )

    # Axes: consistent frame (black box), no grid, no zero line
    fig.update_xaxes(
        showgrid=False,
        zeroline=False,
        showline=True,
        linecolor=frame_color,
        linewidth=frame_linewidth,
        mirror=True,  # top + bottom
        ticks="outside",
        tickfont=dict(size=tick_font_size, color="black"),
        title_font=dict(size=axis_title_size, color="black"),
        automargin=(False if lock_automargins else True),
    )
    fig.update_yaxes(
        showgrid=False,
        zeroline=False,
        showline=True,
        linecolor=frame_color,
        linewidth=frame_linewidth,
        mirror=True,  # left + right
        ticks="outside",
        tickfont=dict(size=tick_font_size, color="black"),
        automargin=(False if lock_automargins else True),
    )

    return fig


def set_log_major_ticks_only(fig: go.Figure, x_values, base=10):
    """
    Force only powers of 10 on the log axis (Option A tick labels),
    removing intermediate ticks like 2 and 5.
    """
    tickvals, ticktext = _major_log_ticks_from_data(x_values, base=base)
    fig.update_xaxes(
        type="log",
        tickmode="array",
        tickvals=tickvals,
        ticktext=ticktext,
        # Keep formatting stable in PDF
        exponentformat="none",
        showexponent="none",
    )
    return fig


# -----------------------------------------------------------------------------
# Your updated functions (minimal changes, but with consistent panel geometry)
# -----------------------------------------------------------------------------
def plot_lcia_boxes_for_reference_product(
    df,
    reference_product,
    title=None,
    yaxis_label=None,
    output_html=None,
    output_image=None,
    image_format="pdf",          # "pdf", "svg", "png"
    image_scale=3,               # only for png
    others_threshold=0.05,
    benchmark_offset=0.3,
    orientation="h",             # "v" or "h"
    include_total=False,
    total_position="first",
    legend_mode="none",          # "full", "none", "only"
    impact_colors=None,
    show_points=True,
    # --- NEW: enforce identical panel geometry
    panel_width=900,
    panel_height=520,
    panel_margins=None,          # dict(l=..., r=..., t=..., b=...)
    tick_font_size=22,
    axis_title_size=20,
    title_size=18,
):
    """
    Same as your version, but:
    - All panels share identical width/height and margins => same black frame size.
    - No Plotly automargins (deterministic geometry across panels).
    """

    ID_COLS = ["Source", "Reference product", "Commodity"]
    impact_cols = [c for c in df.columns if c not in ID_COLS]

    df_rp = df[df["Reference product"] == reference_product].copy()
    if df_rp.empty:
        raise ValueError(f"No data for Reference product = {reference_product}")

    # ---- long format
    df_long = df_rp.melt(
        id_vars=ID_COLS,
        value_vars=impact_cols,
        var_name="Category",
        value_name="Value",
    ).dropna(subset=["Value"])

    # ---- add Total (before Others grouping)
    if include_total:
        totals = (
            df_long.groupby(ID_COLS, as_index=False)["Value"]
            .sum()
            .assign(Category="Total")
        )
        df_long = pd.concat([df_long, totals], ignore_index=True)

    # ---- group small contributors into Others (based on MetalliCan, excluding Total)
    mc = df_long[(df_long["Source"] == "MetalliCan") & (df_long["Category"] != "Total")]
    mc_totals = mc.groupby("Category")["Value"].sum()
    total_impact = mc_totals.sum()

    small_cats = []
    if total_impact and total_impact != 0:
        small_cats = mc_totals[(mc_totals / total_impact) < others_threshold].index.tolist()

    df_long["Category_plot"] = df_long["Category"]
    df_long.loc[df_long["Category"].isin(small_cats), "Category_plot"] = "Others"

    df_long = (
        df_long.groupby(ID_COLS + ["Category_plot"], as_index=False)["Value"]
        .sum()
    )

    # ---- order categories by importance (MetalliCan totals), keep Total position
    order_base = (
        df_long[(df_long["Source"] == "MetalliCan") & (df_long["Category_plot"] != "Total")]
        .groupby("Category_plot", as_index=False)["Value"]
        .sum()
        .sort_values("Value", ascending=False)
    )
    category_order = order_base["Category_plot"].tolist()

    if include_total and "Total" in df_long["Category_plot"].unique():
        category_order = (["Total"] + category_order) if total_position == "first" else (category_order + ["Total"])

    # ensure uniqueness
    seen = set()
    category_order = [c for c in category_order if not (c in seen or seen.add(c))]

    cat_to_pos = {cat: i for i, cat in enumerate(category_order)}
    df_long["pos"] = df_long["Category_plot"].map(cat_to_pos)

    # ---- benchmark medians
    bench = (
        df_long[df_long["Source"].isin(["Ecoinvent", "Regioinvent"])]
        .groupby(["Source", "Category_plot"], as_index=False)["Value"]
        .median()
    )

    # ---- default palette
    if impact_colors is None:
        impact_colors = {
            "Total": "#000000",
            "Climate change": "#d73027",
            "Land transformation": "#1a9850",
            "Marine acidification": "#4575b4",
            "Terrestrial acidification": "#abd9e9",
            "Photochemical ozone": "#fee090",
            "Freshwater ecotoxicity": "#f1b6da",
            "Particulate matter": "#8073ac",
            "Human toxicity non-cancer": "#e08214",
            "Others": "#999999",
        }

    # ---- legend-only output
    if legend_mode == "only":
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(symbol="square", size=16, color="rgba(0,0,0,0)", line=dict(color="black", width=1.5)),
            name="MetalliCan range", showlegend=True,
        ))
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(symbol="circle", size=10, color="rgba(80,80,80,0.7)"),
            name="MetalliCan sites", showlegend=True,
        ))
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(symbol="diamond", size=10, color="#d8b365", line=dict(width=1.6, color="black")),
            name="Ecoinvent", showlegend=True,
        ))
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(symbol="star", size=10, color="#5ab4ac", line=dict(width=1.6, color="black")),
            name="Regioinvent", showlegend=True,
        ))

        fig.update_layout(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            plot_bgcolor="white",
            paper_bgcolor="white",
            margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(
                orientation="h",
                yanchor="middle", y=0.5,
                xanchor="center", x=0.5,
                bgcolor="rgba(255,255,255,0)",
                borderwidth=0,
                font=dict(color="black", size=18),
            ),
            width=panel_width,
            height=int(panel_height * 0.25),
        )

        if output_image:
            fig.write_image(f"{output_image}.{image_format}", scale=image_scale if image_format == "png" else 1)
        if output_html:
            fig.write_html(output_html)
        return fig

    # ---- normal figure
    fig = go.Figure()
    mc_only = df_long[df_long["Source"] == "MetalliCan"].copy()

    for cat in category_order:
        mc_cat = mc_only[mc_only["Category_plot"] == cat]
        if mc_cat.empty:
            continue

        color = impact_colors.get(cat, "#cccccc")

        if orientation == "v":
            x_vals = mc_cat["pos"]
            y_vals = mc_cat["Value"]
        else:
            x_vals = mc_cat["Value"]
            y_vals = mc_cat["pos"]

        fig.add_trace(go.Box(
            x=x_vals,
            y=y_vals,
            boxpoints="all" if show_points else False,
            jitter=0.35 if show_points else 0,
            fillcolor=color,
            line=dict(color="black", width=1),
            marker=dict(color="rgba(80,80,80,0.65)", size=5) if show_points else dict(color="rgba(0,0,0,0)"),
            hovertext=mc_cat["Commodity"],
            hoverinfo="text+x" if orientation == "h" else "text+y",
            showlegend=False,
            orientation="h" if orientation == "h" else "v",
        ))

    # dummy legend entries
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers",
        marker=dict(symbol="square", size=16, color="rgba(0,0,0,0)", line=dict(color="black", width=1.5)),
        name="MetalliCan range",
        showlegend=(legend_mode == "full"),
    ))
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers",
        marker=dict(symbol="circle", size=10, color="rgba(80,80,80,0.7)"),
        name="MetalliCan sites",
        showlegend=(legend_mode == "full"),
    ))

    # benchmarks shifted
    for src, symbol, color, size in [
        ("Ecoinvent", "diamond", "#d8b365", 10),
        ("Regioinvent", "star", "#5ab4ac", 10),
    ]:
        sub = bench[bench["Source"] == src].copy()
        if sub.empty:
            continue

        sub["pos_shift"] = sub["Category_plot"].map(cat_to_pos) - benchmark_offset

        if orientation == "v":
            fig.add_trace(go.Scatter(
                x=sub["pos_shift"], y=sub["Value"],
                mode="markers",
                marker=dict(symbol=symbol, size=size, color=color, line=dict(width=1.6, color="black")),
                name=src,
                showlegend=(legend_mode == "full"),
            ))
        else:
            fig.add_trace(go.Scatter(
                x=sub["Value"], y=sub["pos_shift"],
                mode="markers",
                marker=dict(symbol=symbol, size=size, color=color, line=dict(width=1.6, color="black")),
                name=src,
                showlegend=(legend_mode == "full"),
            ))

    # axis ticks (category axis)
    tick_vals = list(cat_to_pos.values())
    tick_text = list(cat_to_pos.keys())

    if orientation == "v":
        fig.update_xaxes(
            tickmode="array",
            tickvals=tick_vals,
            ticktext=tick_text,
            tickangle=-35,
        )
        fig.update_yaxes(title_text=yaxis_label or "Impact value")
    else:
        fig.update_yaxes(
            tickmode="array",
            tickvals=tick_vals,
            ticktext=tick_text,
            title_text=None,
        )
        fig.update_xaxes(title_text=yaxis_label or "Impact value")

    # Title + legend
    fig.update_layout(
        title=dict(text=title or "", x=0.5),
        showlegend=(legend_mode == "full"),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.20,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="black",
            borderwidth=0.8,
            font=dict(color="black", size=18),
        ) if legend_mode == "full" else dict(),
    )

    # --- APPLY COMMON PANEL GEOMETRY (key for identical black frame size)
    if panel_margins is None:
        # You can tune these once and keep them identical across ALL exports
        panel_margins = dict(l=160, r=20, t=55, b=70)

    apply_common_panel_layout(
        fig,
        width=panel_width,
        height=panel_height,
        margin_l=panel_margins["l"],
        margin_r=panel_margins["r"],
        margin_t=panel_margins["t"],
        margin_b=panel_margins["b"],
        tick_font_size=tick_font_size,
        axis_title_size=axis_title_size,
        title_size=title_size,
        lock_automargins=True,
    )

    if output_html:
        fig.write_html(output_html)

    if output_image:
        fig.write_image(
            f"{output_image}.{image_format}",
            width=panel_width,
            height=panel_height,
            scale=image_scale if image_format == "png" else 1,
        )

    return fig


def plot_lcia_boxes_for_total_damages(
    df,
    reference_products,
    title="",
    xaxis_label="Total impact",
    output_html=None,
    output_image=None,
    image_format="pdf",
    image_scale=3,
    benchmark_offset=0.25,
    show_points=True,
    legend_mode="none",
    order=None,
    # --- NEW: enforce identical panel geometry + major log ticks only
    panel_width=900,
    panel_height=520,
    panel_margins=None,          # dict(l=..., r=..., t=..., b=...)
    tick_font_size=22,
    axis_title_size=20,
    title_size=18,
    log_major_ticks_only=True,   # <-- removes 2 and 5 ticks
):
    """
    Total damages (sum across categories), with strict panel geometry:
    - Same black-frame plot area size as other panels
    - Log axis with ONLY major ticks (Option A)
    """

    ID_COLS = ["Source", "Reference product", "Commodity"]
    impact_cols = [c for c in df.columns if c not in ID_COLS]

    d = df[df["Reference product"].isin(reference_products)].copy()
    if d.empty:
        raise ValueError("No data for selected reference_products.")

    # Long -> compute Total per (Source, Reference product, Commodity)
    d_long = d.melt(
        id_vars=ID_COLS,
        value_vars=impact_cols,
        var_name="Category",
        value_name="Value",
    ).dropna(subset=["Value"])

    d_total = (
        d_long.groupby(["Source", "Reference product", "Commodity"], as_index=False)["Value"]
        .sum()
        .rename(columns={"Value": "Total"})
    )

    # Order on y
    y_order = reference_products if order is None else order
    y_to_pos = {rp: i for i, rp in enumerate(y_order)}
    d_total["y_pos"] = d_total["Reference product"].map(y_to_pos)

    # Bench medians
    bench = (
        d_total[d_total["Source"].isin(["Ecoinvent", "Regioinvent"])]
        .groupby(["Source", "Reference product"], as_index=False)["Total"]
        .median()
    )
    bench["y_pos"] = bench["Reference product"].map(y_to_pos)

    fig = go.Figure()

    # --- MetalliCan boxplots
    mc = d_total[d_total["Source"] == "MetalliCan"].copy()
    for rp in y_order:
        sub = mc[mc["Reference product"] == rp]
        if sub.empty:
            continue

        fig.add_trace(go.Box(
            x=sub["Total"],
            y=[y_to_pos[rp]] * len(sub),
            orientation="h",
            boxpoints="all" if show_points else False,
            jitter=0.35 if show_points else 0,
            fillcolor="rgba(150,150,150,0.35)",
            line=dict(color="black", width=1),
            marker=dict(color="rgba(80,80,80,0.65)", size=5) if show_points else dict(color="rgba(0,0,0,0)"),
            hovertext=sub["Commodity"],
            hoverinfo="text+x",
            showlegend=False,
        ))

    # dummy legend entries
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers",
        marker=dict(symbol="square", size=16, color="rgba(0,0,0,0)", line=dict(color="black", width=1.5)),
        name="MetalliCan range",
        showlegend=(legend_mode == "full"),
    ))
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers",
        marker=dict(symbol="circle", size=10, color="rgba(80,80,80,0.7)"),
        name="MetalliCan sites",
        showlegend=(legend_mode == "full"),
    ))

    # --- Benchmarks (median points)
    for src, symbol, color, size in [
        ("Ecoinvent", "diamond", "#d8b365", 10),
        ("Regioinvent", "star", "#5ab4ac", 10),
    ]:
        sub = bench[bench["Source"] == src].copy()
        if sub.empty:
            continue
        sub["y_shift"] = sub["y_pos"] - benchmark_offset

        fig.add_trace(go.Scatter(
            x=sub["Total"],
            y=sub["y_shift"],
            mode="markers",
            marker=dict(symbol=symbol, size=size, color=color, line=dict(width=1.6, color="black")),
            name=src,
            showlegend=(legend_mode == "full"),
        ))

    # y ticks
    tick_vals = list(y_to_pos.values())
    tick_text = list(y_to_pos.keys())
    fig.update_yaxes(
        tickmode="array",
        tickvals=tick_vals,
        ticktext=tick_text,
    )

    # x axis (log)
    fig.update_xaxes(title_text=xaxis_label)

    if log_major_ticks_only:
        # Use all totals (including benchmarks) to compute the required range
        x_vals_for_ticks = pd.concat([
            d_total["Total"],
            bench["Total"] if "Total" in bench.columns else pd.Series(dtype=float),
        ], ignore_index=True)
        set_log_major_ticks_only(fig, x_vals_for_ticks, base=10)
    else:
        fig.update_xaxes(type="log")

    fig.update_layout(
        title=dict(text=title, x=0.5),
        showlegend=(legend_mode == "full"),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.20,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="black",
            borderwidth=0.8,
            font=dict(color="black", size=18),
        ) if legend_mode == "full" else dict(),
    )

    # --- APPLY COMMON PANEL GEOMETRY (use SAME margins as other function!)
    if panel_margins is None:
        panel_margins = dict(l=160, r=20, t=55, b=70)

    apply_common_panel_layout(
        fig,
        width=panel_width,
        height=panel_height,
        margin_l=panel_margins["l"],
        margin_r=panel_margins["r"],
        margin_t=panel_margins["t"],
        margin_b=panel_margins["b"],
        tick_font_size=tick_font_size,
        axis_title_size=axis_title_size,
        title_size=title_size,
        lock_automargins=True,
    )

    if output_html:
        fig.write_html(output_html)

    if output_image:
        fig.write_image(
            f"{output_image}.{image_format}",
            width=panel_width,
            height=panel_height,
            scale=image_scale if image_format == "png" else 1,
        )

    return fig



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

    # ax.legend(
    #     loc="upper right",
    #     fontsize=16
    #     #bbox_to_anchor=(0.5, -0.15),
    #     #ncol=,
    #     #frameon=False,
    # )

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
            df_ref[df_ref["Source"].isin(["Ecoinvent", "Regioinvent", "MetalliCan market"])]
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
            colors='slategrey',
            linestyles="dotted",
            linewidth=2,
            alpha=0.9,
        )

        if "MetalliCan market" in benchmarks:
            ax.vlines(
            benchmarks["MetalliCan market"],
            y_start - 0.4,
            y_end + 0.4,
            colors='indianred',
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
        Line2D([0], [0], color="slategrey", lw=2, linestyle="dotted", label="Regioinvent"),
        Line2D([0], [0], color="indianred", lw=2, linestyle="dashed", label="MetalliCan market"),
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


# def plot_stacked_contribution_by_activity(
#     df,
#     impact_col="share_%",
#     commodity_col="reference_product_market",
#     name_col="name",
#     flow_type_col="flow_type",
#     commodity_order=None,
#     act_to_hex=None,                     # dict: {activity_name: "#RRGGBB"}
#     act_to_plot=None,                    # list of activities to display (order preserved)
#     top_n=12,                            # used only if act_to_plot=None
#     others_label="Others",
#     hatch_technosphere="///",
#     hatch_biosphere=None,
#     bar_spacing=0.15,
#     figsize=(11, 7),
#     xlabel=None,
#     title=None,
#     normalize_to_percent=False,          # if True: each commodity is normalized to 100%
#     legend=True,
#     legend_fontsize=9,
#     legend_ncol=1,
#     output_path=None,
#     dpi=600,
#     # --- NEW ---
#     direct_first=True,                   # force all Direct (biosphere) to the left
#     legend_direct_first=True,            # force Direct entries first in legend
#     direct_label="",
#     indirect_label=" ",
# ):
#     """
#     Horizontal stacked contribution barplot across ALL commodities.
#
#     - Colors encode activity (name)
#     - Hatch encodes flow_type (Technosphere hatched; Biosphere solid)
#     - Unselected activities grouped into 'Others'
#     - Legend style: "Activity — Direct" and "Activity — Indirect"
#
#     NEW:
#     - direct_first=True: draw ALL Direct segments first (thus leftmost), then Indirect.
#     - legend_direct_first=True: reorder legend so Direct entries appear first.
#
#     Required columns in df:
#       - commodity_col
#       - name_col
#       - flow_type_col (expects e.g. 'Technosphere (first-tier)' and 'Biosphere (direct)')
#       - impact_col
#     """
#     df = df.copy()
#
#     # --- Basic checks
#     for c in [commodity_col, name_col, flow_type_col, impact_col]:
#         if c not in df.columns:
#             raise ValueError(f"Missing column: '{c}'")
#
#     # Coerce impacts to numeric
#     df[impact_col] = pd.to_numeric(df[impact_col], errors="coerce").fillna(0.0)
#
#     # --- Decide which activities to plot
#     if act_to_plot is None:
#         tmp = (
#             df.groupby(name_col, as_index=False)[impact_col].sum()
#             .assign(_abs=lambda x: x[impact_col].abs())
#             .sort_values("_abs", ascending=False)
#         )
#         act_to_plot = tmp[name_col].head(top_n).tolist()
#     act_to_plot = list(act_to_plot)  # preserve given order
#
#     # --- Colors
#     act_to_hex = act_to_hex or {}
#
#     def _get_color(act, i):
#         if act in act_to_hex:
#             return act_to_hex[act]
#         cmap = plt.get_cmap("tab20")
#         return cmap(i % 20)
#
#     # --- Group non-selected -> Others
#     df["_act"] = df[name_col].where(df[name_col].isin(act_to_plot), others_label)
#
#     # --- Aggregate
#     agg = (
#         df.groupby([commodity_col, "_act", flow_type_col], as_index=False)[impact_col]
#         .sum()
#         .rename(columns={impact_col: "impact"})
#     )
#
#     # --- Commodity order
#     if commodity_order is not None:
#         missing = set(commodity_order) - set(agg[commodity_col].unique())
#         if missing:
#             raise ValueError(f"Commodities not found in data: {missing}")
#     else:
#         commodity_order = (
#             agg.groupby(commodity_col, as_index=False)["impact"].sum()
#             .assign(_abs=lambda x: x["impact"].abs())
#             .sort_values("_abs", ascending=False)[commodity_col]
#             .tolist()
#         )
#
#     # --- Fixed bar heights
#     heights = np.full(len(commodity_order), 0.7, dtype=float)
#
#     # y positions with spacing
#     y_pos = np.zeros(len(commodity_order), dtype=float)
#     for i in range(1, len(commodity_order)):
#         y_pos[i] = y_pos[i - 1] + heights[i - 1] + bar_spacing
#
#     # --- Flow labels (expected)
#     flow_bio = "Biosphere (direct)"
#     flow_tech = "Technosphere (first-tier)"
#
#     # --- Flow order: bio first, then tech, then any others
#     flow_types_present = set(agg[flow_type_col].unique().tolist())
#     flow_order = []
#     if flow_bio in flow_types_present:
#         flow_order.append(flow_bio)
#     if flow_tech in flow_types_present:
#         flow_order.append(flow_tech)
#     flow_order += [ft for ft in sorted(flow_types_present) if ft not in set(flow_order)]
#
#     # --- Activity order
#     acts_present = set(agg["_act"].unique().tolist())
#     act_order = [a for a in act_to_plot if a in acts_present]
#     if others_label in acts_present:
#         act_order.append(others_label)
#
#     # --- Pivot
#     pivot = (
#         agg.pivot_table(
#             index=[commodity_col],
#             columns=["_act", flow_type_col],
#             values="impact",
#             aggfunc="sum",
#             fill_value=0.0,
#         )
#         .reindex(index=commodity_order, fill_value=0.0)
#     )
#
#     # Normalize each commodity to 100% if requested
#     if normalize_to_percent:
#         row_sums = pivot.sum(axis=1).replace(0, np.nan)
#         pivot = (pivot.div(row_sums, axis=0) * 100.0).fillna(0.0)
#         if xlabel is None:
#             xlabel = "Contribution (%)"
#     else:
#         if xlabel is None:
#             xlabel = impact_col
#
#     # --- Plot
#     fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
#     left = np.zeros(len(commodity_order), dtype=float)
#
#     labeled = set()
#
#     def _direct_or_indirect(ft: str) -> str:
#         # Map known flow types to Direct/Indirect; fallback: treat as Indirect
#         if ft == flow_bio:
#             return direct_label
#         if ft == flow_tech:
#             return indirect_label
#         return indirect_label
#
#     def _hatch_for(ft: str):
#         if ft == flow_tech:
#             return hatch_technosphere
#         if ft == flow_bio:
#             return hatch_biosphere
#         # unknown: treat like indirect hatch (safer visual cue)
#         return hatch_technosphere
#
#     # --- Drawing order:
#     # If direct_first: draw ALL flows in flow_order outer loop (Direct first), so Direct is leftmost.
#     # Else: draw by activity first, then flow (original behavior).
#     if direct_first:
#         outer = [("flow", ft) for ft in flow_order]
#         inner_kind = "act"
#     else:
#         outer = [("act", act) for act in act_order]
#         inner_kind = "flow"
#
#     def _iter_pairs():
#         if direct_first:
#             for ft in flow_order:
#                 for i_act, act in enumerate(act_order):
#                     yield act, i_act, ft
#         else:
#             for i_act, act in enumerate(act_order):
#                 for ft in flow_order:
#                     yield act, i_act, ft
#
#     for act, i_act, ft in _iter_pairs():
#         color = _get_color(act, i_act)
#
#         try:
#             vals = pivot[(act, ft)].values
#         except KeyError:
#             vals = np.zeros(len(commodity_order), dtype=float)
#
#         if np.allclose(vals, 0):
#             continue
#
#         hatch = _hatch_for(ft)
#
#         suffix = _direct_or_indirect(ft)
#         label_full = f"{act} {suffix}"
#         label = label_full if label_full not in labeled else "_nolegend_"
#         if label != "_nolegend_":
#             labeled.add(label_full)
#
#         ax.barh(
#             y_pos,
#             vals,
#             left=left,
#             height=heights,
#             color=color,
#             hatch=hatch,
#             edgecolor="black",
#             linewidth=0.5,
#             label=label,
#         )
#         left += vals
#
#     # Axes
#     ax.set_yticks(y_pos)
#     ax.set_yticklabels(commodity_order)
#     ax.invert_yaxis()
#     ax.set_xlabel(xlabel, fontsize=12)
#     ax.tick_params(axis="x", labelsize=11)
#     ax.tick_params(axis="y", labelsize=11)
#
#     if title is not None:
#         ax.set_title(title, fontsize=12)
#
#     # --- Legend: force Direct entries first (then Indirect)
#     if legend:
#         handles, labels = ax.get_legend_handles_labels()
#         pairs = [(h, l) for h, l in zip(handles, labels) if l and l != "_nolegend_"]
#
#         if legend_direct_first:
#             def _is_direct(lbl: str) -> bool:
#                 return f"— {direct_label}" in lbl
#
#             # Direct first; stable sort keeps within-group order
#             pairs = sorted(pairs, key=lambda hl: (0 if _is_direct(hl[1]) else 1))
#
#         handles_sorted = [h for h, _ in pairs]
#         labels_sorted = [l for _, l in pairs]
#
#         ax.legend(
#             handles_sorted,
#             labels_sorted,
#             loc="upper left",
#             bbox_to_anchor=(1.01, 1.0),
#             frameon=False,
#             fontsize=legend_fontsize,
#             ncol=legend_ncol,
#         )
#
#     # Save
#     if output_path:
#         fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
#
#     plt.show()
#     return fig, ax


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
    legend_fontsize=12,
    legend_ncol=1,                       # (kept for backward compat; not used in split legend mode)
    output_path=None,
    dpi=600,
    # --- NEW ---
    direct_first=True,                   # force all Direct (biosphere) to the left
    legend_direct_first=True,            # (kept for backward compat; not used in split legend mode)
    direct_label="Direct",
    indirect_label="Indirect",
    # --- NEW legend control ---
    legend_below=True,                   # place legend below plot
    legend_split=True,                   # split into Direct vs Indirect blocks
    legend_titles=("Direct impacts", "Indirect impacts"),
    legend_cols_each=1,                  # columns inside each legend block
    legend_y=-0.14,                      # vertical anchor (negative = below axes)
    legend_bottom_margin=0.25,           # space reserved at bottom for legends
):
    """
    Horizontal stacked contribution barplot across ALL commodities.

    - Colors encode activity (name)
    - Hatch encodes flow_type (Technosphere hatched; Biosphere solid)
    - Unselected activities grouped into 'Others'

    NEW:
    - direct_first=True: draw ALL Direct segments first (thus leftmost), then Indirect.
    - legend_below + legend_split: put two legend blocks below (Direct solid / Indirect hatched).
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
    act_to_plot = list(act_to_plot)  # preserve order

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

    # --- Fixed bar heights
    heights = np.full(len(commodity_order), 0.7, dtype=float)

    # y positions with spacing
    y_pos = np.zeros(len(commodity_order), dtype=float)
    for i in range(1, len(commodity_order)):
        y_pos[i] = y_pos[i - 1] + heights[i - 1] + bar_spacing

    # --- Flow labels (expected)
    flow_bio = "Biosphere (direct)"
    flow_tech = "Technosphere (first-tier)"

    # --- Flow order: bio first, then tech, then any others
    flow_types_present = set(agg[flow_type_col].unique().tolist())
    flow_order = []
    if flow_bio in flow_types_present:
        flow_order.append(flow_bio)
    if flow_tech in flow_types_present:
        flow_order.append(flow_tech)
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

    # --- Plot
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=False)
    left = np.zeros(len(commodity_order), dtype=float)

    # stable color per activity (for legend proxies)
    act_colors = {act: _get_color(act, i) for i, act in enumerate(act_order)}

    def _hatch_for(ft: str):
        if ft == flow_tech:
            return hatch_technosphere
        if ft == flow_bio:
            return hatch_biosphere
        return hatch_technosphere

    def _iter_pairs():
        if direct_first:
            for ft in flow_order:
                for i_act, act in enumerate(act_order):
                    yield act, i_act, ft
        else:
            for i_act, act in enumerate(act_order):
                for ft in flow_order:
                    yield act, i_act, ft

    for act, i_act, ft in _iter_pairs():
        color = act_colors[act]

        try:
            vals = pivot[(act, ft)].values
        except KeyError:
            vals = np.zeros(len(commodity_order), dtype=float)

        if np.allclose(vals, 0):
            continue

        hatch = _hatch_for(ft)

        ax.barh(
            y_pos,
            vals,
            left=left,
            height=heights,
            color=color,
            hatch=hatch,
            edgecolor="black",
            linewidth=0.5,
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

    # --- Legend: below, centered, split in 2 blocks (Direct solid vs Indirect hatched)
    if legend and legend_below and legend_split:
        present_direct = []
        present_indirect = []

        for act in act_order:
            # direct = biosphere (solid)
            try:
                v_dir = pivot[(act, flow_bio)].values
            except KeyError:
                v_dir = np.zeros(len(commodity_order))
            if not np.allclose(v_dir, 0):
                present_direct.append(act)

            # indirect = technosphere (hatched)
            try:
                v_ind = pivot[(act, flow_tech)].values
            except KeyError:
                v_ind = np.zeros(len(commodity_order))
            if not np.allclose(v_ind, 0):
                present_indirect.append(act)

        direct_handles = [
            Patch(
                facecolor=act_colors.get(act, "0.8"),
                edgecolor="black",
                linewidth=0.5,
                hatch=None,
                label=act,
            )
            for act in present_direct
        ]
        indirect_handles = [
            Patch(
                facecolor=act_colors.get(act, "0.8"),
                edgecolor="black",
                linewidth=0.5,
                hatch=hatch_technosphere,
                label=act,
            )
            for act in present_indirect
        ]

        # reserve space at bottom for the legends
        fig.subplots_adjust(bottom=legend_bottom_margin)

        # left block: Direct
        leg1 = ax.legend(
            handles=direct_handles,
            title=legend_titles[0],
            loc="upper center",
            bbox_to_anchor=(0.25, legend_y),
            frameon=False,
            fontsize=legend_fontsize,
            title_fontsize=legend_fontsize,
            ncol=legend_cols_each,
            handlelength=1.6,
            handleheight=1.0,
            handletextpad=0.6,
            columnspacing=1.2,
        )
        ax.add_artist(leg1)

        # right block: Indirect
        ax.legend(
            handles=indirect_handles,
            title=legend_titles[1],
            loc="upper center",
            bbox_to_anchor=(0.75, legend_y),
            frameon=False,
            fontsize=legend_fontsize,
            title_fontsize=legend_fontsize,
            ncol=legend_cols_each,
            handlelength=1.6,
            handleheight=1.0,
            handletextpad=0.6,
            columnspacing=1.2,
        )

    # Fallback: original side legend (kept simple)
    elif legend:
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(
            handles,
            labels,
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
    return fig, ax
