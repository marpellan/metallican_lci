import bw2analyzer as ba
import bw2calc as bc
import bw2data as bd
import bw2io as bi
import brightway2 as bw
import pandas as pd
import numpy as np
import datetime
import math # for pedigree matrix
import re


def get_inventory_dataset(inventories, database_names):
    """
    Function from Istrate et al (2024) to find the dataset in the specified databases.

    :param inventories: dict in the format (mineral name: activity name, reference product, location)
    :param database_names: must be a list
    :return df:
    """
    inventory_ds = {}
    for rm_name, (activity_name, ref_product, location) in inventories.items():
        match_found = False

        # Iterate over the list of database names
        for database_name in database_names:
            db = bw.Database(database_name)
            matches = [ds for ds in db if ds["name"] == activity_name
                       and ds["reference product"] == ref_product
                       and ds["location"] == location]

            if matches:
                inventory_ds[rm_name] = matches[0]
                match_found = True
                break  # Stop searching once a match is found

        if not match_found:
            print(f"No match found for {rm_name} in provided databases")
    return inventory_ds


def run_lca(inventories, amount, lcia_methods):
    """
    Compute LCA scores for multiple inventories and multiple methods.

    Returns a DataFrame indexed by inventory label, columns = "Impact name (unit)".
    """
    # canonicalize methods dict
    if isinstance(lcia_methods, list):
        # expect tuples like (method_id, ..., label)
        lcia_methods = {tpl[2]: tpl[0] for tpl in lcia_methods}
    if not isinstance(lcia_methods, dict):
        raise TypeError("lcia_methods must be dict or list of tuples")

    results = {}
    for label, activity in inventories.items():
        # one LCA object per activity
        lca = bw.LCA({activity.key: amount})
        lca.lci()
        row = {}
        for imp_label, method_id in lcia_methods.items():
            lca.switch_method(method_id)
            lca.lcia()
            unit = bw.Method(method_id).metadata.get("unit", "")
            row[f"{imp_label} ({unit})"] = lca.score
        results[label] = row

    df = pd.DataFrame.from_dict(results, orient="index")
    df.index.name = "Commodity"
    return df.reset_index()


def compute_mp_contributions_to_ep(
    inventories,
    amount=1,
    damage_version="IMPACT World+ Damage 2.1_regionalized for ecoinvent v3.10"
):
    """
    For each activity in `inventories`, computes midpoint→endpoint damage shares

    Returns
    -------
    df_hh : pd.DataFrame
        rows = Commodity, columns = each Human health midpoint, values = share of total HH damage
    df_eq : pd.DataFrame
        rows = Commodity, columns = each Ecosystem quality midpoint, values = share of total EQ damage
    """
    # 1) grab all triples for that damage version
    all_damage = [m for m in bw.methods if m[0] == damage_version]

    # 2) identify endpoint / midpoint lists per category
    categories = ["Human health", "Ecosystem quality"]
    endpoints = {
        cat: next(
            m for m in all_damage
            if m[1] == cat and m[2].lower().startswith("total")
        )
        for cat in categories
    }
    midpoints = {
        cat: [
            m for m in all_damage
            if m[1] == cat and not m[2].lower().startswith("total")
        ]
        for cat in categories
    }

    # 3) loop over inventories, fill dicts
    hh_shares = {}
    eq_shares = {}

    for comm, activity in inventories.items():
        lca = bw.LCA({activity.key: amount})
        lca.lci()

        # temporary storage per commodity
        tmp = {}

        for cat in categories:
            # compute total
            lca.switch_method(endpoints[cat])
            lca.lcia()
            total = lca.score or 1e-30

            # compute each midpoint share
            shares = {}
            for mid in midpoints[cat]:
                lca.switch_method(mid)
                lca.lcia()
                shares[mid[2]] = lca.score / total

            tmp[cat] = shares

        hh_shares[comm] = tmp["Human health"]
        eq_shares[comm] = tmp["Ecosystem quality"]

    # 4) build DataFrames
    df_hh = pd.DataFrame.from_dict(hh_shares, orient="index")\
             .reset_index()\
             .rename(columns={"index": "Commodity"})

    df_eq = pd.DataFrame.from_dict(eq_shares, orient="index")\
             .reset_index()\
             .rename(columns={"index": "Commodity"})

    return df_hh, df_eq



def direct_biosphere_contributions(activity, method_id, amount=1.0, threshold=0.0):
    """
    Direct elementary-flow (biosphere) contributions of the *foreground activity only*.
    Version-safe across Brightway 2.x variants.

    Parameters
    ----------
    activity : bw.Activity
        Foreground activity object
    method_id : tuple
        Brightway method key, e.g. ("IPCC 2021", "climate change", "GWP 100a")
    amount : float
        Functional unit amount (in activity reference unit)
    threshold : float
        Minimum share (0-1) of total score to keep

    Returns
    -------
    pd.DataFrame
    """
    # Run LCA once (gives total score and scaling)
    lca = bw.LCA({activity.key: amount}, method_id)
    lca.lci()
    lca.lcia()

    total = float(lca.score) if lca.score else 1e-30

    # ---- Get scaling factor for the foreground activity in solved system ----
    # Try common attribute names, fallback to reverse_dict if needed.
    scale = None

    if hasattr(lca, "activity_dict"):  # common in bw2calc
        idx = lca.activity_dict[activity.key]
        scale = float(lca.supply_array[idx])

    else:
        # fallback: reverse_dict() gives idx -> key mapping, so invert it
        ra, rp, rb = lca.reverse_dict()
        inv_ra = {v: k for k, v in ra.items()}  # key -> idx
        idx = inv_ra[activity.key]
        scale = float(lca.supply_array[idx])

    # ---- CF lookup (flow key -> CF) ----
    cfs = dict(bw.Method(method_id).load())

    rows = []
    for exc in activity.biosphere():
        flow = exc.input
        cf = float(cfs.get(flow.key, 0.0))
        if cf == 0.0:
            continue

        impact = float(exc["amount"]) * scale * cf
        share = impact / total

        if share < threshold:
            continue

        rows.append({
            "Flow": flow.get("name"),
            "Categories": flow.get("categories"),
            "Unit": flow.get("unit"),
            "Direct_amount_scaled": float(exc["amount"]) * scale,
            "CF": cf,
            "Impact_score": impact,
            "Share_%": 100 * share,
        })

    return (
        pd.DataFrame(rows)
        .sort_values("Share_%", ascending=False)
        .reset_index(drop=True)
    )

def first_tier_contributions(activity, method_id, amount=1.0, threshold=0.0, top=None):
    """
    First-tier technosphere contribution analysis:
    - considers ONLY direct technosphere inputs of `activity`
    - attributes impact as: (required amount of supplier) * (LCIA score of supplier per 1 unit)

    Shares are relative to the TOTAL LCIA score of the full system of the foreground activity.

    Version-safe across Brightway 2.x variants (no lca.dicts usage).
    """
    # --- Foreground LCA (for total score + scaling of foreground activity) ---
    lca_fg = bw.LCA({activity.key: amount}, method_id)
    lca_fg.lci()
    lca_fg.lcia()

    total = float(lca_fg.score) if lca_fg.score else 1e-30

    # --- Get scaling factor for the foreground activity in solved system ---
    if hasattr(lca_fg, "activity_dict"):
        idx = lca_fg.activity_dict[activity.key]
        scale = float(lca_fg.supply_array[idx])
    else:
        ra, rp, rb = lca_fg.reverse_dict()
        inv_ra = {v: k for k, v in ra.items()}  # key -> idx
        idx = inv_ra[activity.key]
        scale = float(lca_fg.supply_array[idx])

    rows = []

    # --- Loop over direct technosphere exchanges (first tier) ---
    for exc in activity.technosphere():
        supplier = exc.input

        # required amount of supplier in the solved foreground system
        req_amount = float(exc["amount"]) * scale

        # supplier LCIA per 1 unit of its reference product
        lca_sup = bw.LCA({supplier.key: 1.0}, method_id)
        lca_sup.lci()
        lca_sup.lcia()
        sup_score = float(lca_sup.score) if lca_sup.score else 0.0

        impact = req_amount * sup_score
        share = impact / total if total else 0.0

        if share < threshold:
            continue

        rows.append({
            "Supplier_name": supplier.get("name"),
            "Supplier_ref_product": supplier.get("reference product"),
            "Supplier_location": supplier.get("location"),
            "Supplier_unit": supplier.get("unit"),
            "Exchange_amount_per_FU": req_amount,   # already scaled to your FU
            "Supplier_score_per_unit": sup_score,   # impact per 1 unit of supplier
            "Impact_score": impact,
            "Share_%": 100 * share,
        })

    df = (
        pd.DataFrame(rows)
        .sort_values("Share_%", ascending=False)
        .reset_index(drop=True)
    )

    if top is not None:
        df = df.head(top)

    return df

# def _infer_commodity_from_label(label: str):
#     """
#     Heuristic:
#     - 'Gibraltar (Cu)' -> 'Cu'
#     - 'Snow Lake (Zn)' -> 'Zn'
#     - 'Goldex (Au)' -> 'Au'
#     - 'LaRonde (Doré)' -> 'Doré'
#     - If no parentheses, return None
#     """
#     m = re.search(r"\(([^)]+)\)\s*$", str(label))
#     return m.group(1).strip() if m else None


def direct_bio_and_first_tier_tech(
    inventory_dict,
    method_id,
    amount=1.0,
    threshold=0.0,          # share threshold (0–1)
    top=None,               # top N per (lci_name, flow_type)
    cache_supplier_scores=True,
    supplier_score_cache=None,
):
    """
    For each {lci_name: activity} in inventory_dict:
      - Direct biosphere contributions (foreground activity only)
      - First-tier technosphere contributions (direct inputs only)

    Returns a single DataFrame with:
      ['lci_name', 'activity_name', 'flow_type',
       'name', 'reference_product', 'categories',
       'unit', 'location',
       'impact_score', 'share_%', 'total_score']
    """

    if not isinstance(inventory_dict, dict) or len(inventory_dict) == 0:
        raise ValueError("inventory_dict must be a non-empty dict: {lci_name: bw_activity}")

    if supplier_score_cache is None:
        supplier_score_cache = {}

    rows_all = []

    # Preload CFs once
    cfs = dict(bw.Method(method_id).load())

    for lci_name, activity in inventory_dict.items():

        # --- Foreground LCA once ---
        lca_fg = bw.LCA({activity.key: amount}, method_id)
        lca_fg.lci()
        lca_fg.lcia()

        total = float(lca_fg.score) if lca_fg.score else 1e-30

        # Version-safe scaling
        if hasattr(lca_fg, "activity_dict"):
            idx = lca_fg.activity_dict[activity.key]
            scale = float(lca_fg.supply_array[idx])
        else:
            ra, rp, rb = lca_fg.reverse_dict()
            inv_ra = {v: k for k, v in ra.items()}
            idx = inv_ra[activity.key]
            scale = float(lca_fg.supply_array[idx])

        activity_name = activity.get("name")

        # =========================
        # 1) Direct biosphere
        # =========================
        for exc in activity.biosphere():
            flow = exc.input
            cf = float(cfs.get(flow.key, 0.0))
            if cf == 0.0:
                continue

            impact = float(exc["amount"]) * scale * cf
            share = impact / total

            if share < threshold:
                continue

            rows_all.append({
                "lci_name": lci_name,
                "activity_name": activity_name,
                "flow_type": "Biosphere (direct)",
                "name": flow.get("name"),
                "reference_product": None,
                "categories": flow.get("categories"),
                "unit": flow.get("unit"),
                "location": None,
                "impact_score": impact,
                "share_%": 100 * share,
                "total_score": total,
            })

        # =========================
        # 2) First-tier technosphere
        # =========================
        for exc in activity.technosphere():
            supplier = exc.input
            req_amount = float(exc["amount"]) * scale

            sup_key = supplier.key
            if cache_supplier_scores and sup_key in supplier_score_cache:
                sup_score = supplier_score_cache[sup_key]
            else:
                lca_sup = bw.LCA({supplier.key: 1.0}, method_id)
                lca_sup.lci()
                lca_sup.lcia()
                sup_score = float(lca_sup.score) if lca_sup.score else 0.0
                if cache_supplier_scores:
                    supplier_score_cache[sup_key] = sup_score

            impact = req_amount * sup_score
            share = impact / total

            if share < threshold:
                continue

            rows_all.append({
                "lci_name": lci_name,
                "activity_name": activity_name,
                "flow_type": "Technosphere (first-tier)",
                "name": supplier.get("name"),
                "reference_product": supplier.get("reference product"),
                "categories": None,
                "unit": supplier.get("unit"),
                "location": supplier.get("location"),
                "impact_score": impact,
                "share_%": 100 * share,
                "total_score": total,
            })

    df = pd.DataFrame(rows_all)

    if df.empty:
        return pd.DataFrame(columns=[
            "lci_name","activity_name","flow_type","name",
            "reference_product","categories","unit","location",
            "impact_score","share_%","total_score"
        ])

    # Sort within each LCI
    df = (
        df.sort_values(["lci_name", "flow_type", "share_%"],
                       ascending=[True, True, False])
          .reset_index(drop=True)
    )

    # Optional: keep top N per LCI & flow type
    if top is not None:
        df = (
            df.groupby(["lci_name", "flow_type"], group_keys=False)
              .head(top)
              .reset_index(drop=True)
        )

    return df


def weighted_market_contributions(
    contribution_df: pd.DataFrame,
    market_shares_df: pd.DataFrame,
    join_cols=("lci_name", "activity_name"),
    market_refprod_col="reference_product",   # in market_shares_df: Doré, Cu concentrate, U concentrate...
    share_col="market_share",
    normalize_shares=True,
    eps=1e-30,
):
    """
    Build a market-level contribution analysis as a weighted mix of site-level LCAs.

    contribution_df: output of your site contribution function, with at least:
      ['lci_name','activity_name','flow_type','name','reference_product','categories','unit','location',
       'impact_score','share_%','total_score']

      NOTE: contribution_df['reference_product'] = supplier reference product (technosphere rows),
            not the market reference product. We'll rename it to supplier_reference_product.

    market_shares_df must contain:
      [market_refprod_col, share_col] + join_cols

    Returns:
      df_market: market contribution table (same idea as site table)
      df_market_total: market total score per reference product
      df_joined: joined site rows (debug)
    """

    # Rename to avoid name collision: contribution reference_product = supplier reference product
    dfc = contribution_df.copy().rename(columns={"reference_product": "supplier_reference_product"})
    dfs = market_shares_df.copy()

    # Normalize shares PER reference_product on the market_shares table (not after merge!)
    if normalize_shares:
        s = dfs.groupby(market_refprod_col)[share_col].transform("sum")
        dfs["_share_norm"] = np.where(s > 0, dfs[share_col] / s, dfs[share_col])
    else:
        dfs["_share_norm"] = dfs[share_col]

    # Join: attach market ref product + weights to each site-level contribution row
    df = dfc.merge(
        dfs[list(join_cols) + [market_refprod_col, "_share_norm"]],
        on=list(join_cols),
        how="inner",
    )
    if df.empty:
        raise ValueError("Join produced 0 rows. Check that join_cols match in both dataframes.")

    # Weighted impact for each contribution item
    df["weighted_impact"] = df["impact_score"] * df["_share_norm"]

    # --- Market total score per reference product: sum_i w_i * total_score_i ---
    # total_score is repeated per contribution row => keep one per site
    tot_site = df[[market_refprod_col] + list(join_cols) + ["total_score", "_share_norm"]].drop_duplicates()
    tot_site["weighted_total"] = tot_site["total_score"] * tot_site["_share_norm"]

    df_market_total = (
        tot_site.groupby(market_refprod_col)
        .agg(
            total_score_market=("weighted_total", "sum"),
            n_sites=(join_cols[0], "nunique"),
            sum_shares=("_share_norm", "sum"),
        )
        .reset_index()
        .rename(columns={market_refprod_col: "reference_product_market"})
    )

    # --- Market contributions: sum_i w_i * impact_score_{i,k} ---
    df = df.rename(columns={market_refprod_col: "reference_product_market"})

    item_cols = [
        "reference_product_market",
        "flow_type",
        "name",
        "supplier_reference_product",
        "categories",
        "unit",
        "location",
    ]

    df_market = (
        df.groupby(item_cols, dropna=False)
        .agg(
            impact_score_market=("weighted_impact", "sum"),
            n_sites=("lci_name", "nunique"),
        )
        .reset_index()
        .merge(
            df_market_total[["reference_product_market", "total_score_market"]],
            on="reference_product_market",
            how="left",
        )
    )

    # Recompute shares relative to the MARKET total (this is what you want)
    df_market["share_%"] = 100 * df_market["impact_score_market"] / df_market["total_score_market"].replace(0, eps)

    df_market = df_market.sort_values(
        ["reference_product_market", "flow_type", "share_%"],
        ascending=[True, True, False],
    ).reset_index(drop=True)

    return df_market, df_market_total, df


def aggregate_electricity_market(
    df,
    activity_pattern="market for electricity, medium voltage",
    location_value="CA",
):
    df = df.copy()

    # 1) Filtrer les lignes électricité
    mask = df["name"].str.contains(activity_pattern, case=False, na=False)
    df_elec = df[mask]
    df_other = df[~mask]

    if df_elec.empty:
        return df  # rien à agréger

    # 2) Colonnes numériques à sommer (SAUF total_score_market)
    num_cols = df.select_dtypes(include="number").columns.tolist()
    sum_cols = [c for c in num_cols if c != "total_score_market"]

    # 3) Agrégation par reference_product_market
    agg_sum = df_elec.groupby("reference_product_market", as_index=False)[sum_cols].sum()

    # total_score_market: on le garde tel quel (1 valeur par reference_product_market)
    total = (
        df_elec.groupby("reference_product_market", as_index=False)["total_score_market"]
        .first()
    )

    df_elec_agg = agg_sum.merge(total, on="reference_product_market", how="left")

    # 4) Remettre les colonnes non-numériques (valeurs constantes cohérentes)
    df_elec_agg["flow_type"] = df_elec["flow_type"].dropna().iloc[0] if df_elec["flow_type"].notna().any() else None
    df_elec_agg["supplier_reference_product"] = df_elec["supplier_reference_product"].dropna().iloc[0] if df_elec["supplier_reference_product"].notna().any() else None
    df_elec_agg["unit"] = df_elec["unit"].dropna().iloc[0] if df_elec["unit"].notna().any() else None
    df_elec_agg["categories"] = df_elec["categories"].dropna().iloc[0] if df_elec["categories"].notna().any() else None

    df_elec_agg["name"] = activity_pattern
    df_elec_agg["location"] = location_value

    # 5) Réordonner SANS KeyError (reindex crée les colonnes manquantes si besoin)
    df_elec_agg = df_elec_agg.reindex(columns=df.columns)

    # 6) Recombiner
    return pd.concat([df_other, df_elec_agg], ignore_index=True)


def export_activity_exchanges(inventory_ds, output_folder="exports"):
    """
    Export technosphere and biosphere flows for each activity in inventory_ds.
    """
    import os
    os.makedirs(output_folder, exist_ok=True)

    all_rows = []  # to combine all results if needed

    for rm_name, act in inventory_ds.items():
        tech_exchanges = []
        bio_exchanges = []

        # --- Technosphere ---
        for exc in act.technosphere():
            tech_exchanges.append({
                "raw_material": rm_name,
                "activity_name": act["name"],
                "reference_product": act["reference product"],
                "location": act["location"],
                "exchange_type": "technosphere",
                "input_name": exc.input["name"],
                "input_database": exc.input["database"],
                "input_code": exc.input["code"],
                "amount": exc.amount,
                "unit": exc.input.get("unit", None),
                "comment": exc.get("comment", None)
            })

        # --- Biosphere ---
        for exc in act.biosphere():
            bio_exchanges.append({
                "raw_material": rm_name,
                "activity_name": act["name"],
                "reference_product": act["reference product"],
                "location": act["location"],
                "exchange_type": "biosphere",
                "input_name": exc.input["name"],
                "input_database": exc.input["database"],
                "input_code": exc.input["code"],
                "amount": exc.amount,
                "unit": exc.input.get("unit", None),
                "comment": exc.get("comment", None)
            })

        # --- Export individual CSVs ---
        pd.DataFrame(tech_exchanges).to_csv(
            f"{output_folder}/{rm_name}_technosphere.csv", index=False
        )
        pd.DataFrame(bio_exchanges).to_csv(
            f"{output_folder}/{rm_name}_biosphere.csv", index=False
        )

        all_rows.extend(tech_exchanges + bio_exchanges)

    # --- Optional: export all combined ---
    pd.DataFrame(all_rows).to_csv(f"{output_folder}/all_exchanges.csv", index=False)
    print(f"✅ Export complete: CSVs saved in '{output_folder}/'")


# def create_pedigree_matrix(pedigree_scores: tuple, exc_amount: float):
#     """
#     Function from Istrate et al (2024)
#
#     This function returns a dict containing the pedigree matrix dict and loc and scale values
#     that can be used to update exchanges in a dataset dict
#
#     The pedigree matrix dictionary is created using the scores provided in the LCI Excel file.
#
#     The code to calcualte the loc and scale values is based on https://github.com/brightway-lca/pedigree_matrix,
#     which is published by Chris Mutel under an BSD 3-Clause License (2021).
#
#     :param pedigree_scores: tuple of pedigree scores
#     :param exc_amount: exchange amount
#     :return dict:
#     """
#
#     VERSION_2 = {
#         "reliability": (1.0, 1.54, 1.61, 1.69, 1.69),
#         "completeness": (1.0, 1.03, 1.04, 1.08, 1.08),
#         "temporal correlation": (1.0, 1.03, 1.1, 1.19, 1.29),
#         "geographical correlation": (1.0, 1.04, 1.08, 1.11, 1.11),
#         "further technological correlation": (1.0, 1.18, 1.65, 2.08, 2.8),
#         "sample size": (1.0, 1.0, 1.0, 1.0, 1.0),
#     }
#
#     pedigree_scores_dict = {
#         'reliability': pedigree_scores[0],
#         'completeness': pedigree_scores[1],
#         'temporal correlation': pedigree_scores[2],
#         'geographical correlation': pedigree_scores[3],
#         'further technological correlation': pedigree_scores[4]
#     }
#
#     assert len(pedigree_scores) in (5, 6), "Must provide either 5 or 6 factors"
#     if len(pedigree_scores) == 5:
#         pedigree_scores = pedigree_scores + (1,)
#
#     factors = [VERSION_2[key][index - 1] for key, index in pedigree_scores_dict.items()]
#
#     basic_uncertainty: float = 1.0
#     values = [basic_uncertainty] + factors
#
#     scale = math.sqrt(sum([math.log(x) ** 2 for x in values])) / 2
#     loc = math.log(abs(exc_amount))
#
#     pedigree_dict = {
#         'uncertainty type': 2,
#         'loc': loc,
#         'scale': scale,
#         "pedigree": pedigree_scores_dict,
#     }
#     return pedigree_dict


### LCI ###
# def search_activity(database_name, activity_name, ref_product, location):
#     """
#     Function to find a specific activity based on its name and reference product in a BW database
#     """
#     db = bw.Database(database_name)
#     matches = [ds for ds in db if ds["name"] == activity_name
#                and ds["reference product"] == ref_product
#                and ds["location"] == location]
#
#     if matches:
#         print(f"Match found in {database_name}:")
#         for match in matches:
#             print(match)
#     else:
#         print(
#             f"No match found in {database_name} for activity '{activity_name}', product '{ref_product}', location '{location}'")
#
#
# def filter_ecoinvent_activities(databases_to_include, products_to_include, locations_to_include=None):
#     """
#     Extracts activities from Ecoinvent databases based on product and location filters.
#
#     Parameters:
#     - databases_to_include (list): List of Ecoinvent databases to search.
#     - products_to_include (list): List of product/activity keywords to match.
#     - locations_to_include (list, optional): List of locations to filter (default: None, includes all locations).
#
#     Returns:
#     - pd.DataFrame: DataFrame with filtered Ecoinvent activities.
#     """
#     data = []
#
#     for db_name in databases_to_include:
#         if db_name in bw.databases:  # Check if the database exists
#             db = bw.Database(db_name)
#             for activity in db:
#                 product_name = activity.get('reference product', None)
#                 activity_name = activity['name']
#                 location = activity.get('location', None)
#
#                 # Match product keyword in either 'reference product' or 'activity name'
#                 matched_metal = next(
#                     (p for p in products_to_include if
#                      (product_name and re.search(rf'\b{p}\b', product_name, re.IGNORECASE)) or
#                      (activity_name and re.search(rf'\b{p}\b', activity_name, re.IGNORECASE))
#                     ),
#                     None
#                 )
#
#                 # Apply location filter if specified
#                 if matched_metal and (locations_to_include is None or location in locations_to_include):
#                     data.append({
#                         'metal': matched_metal,  # Add the identified metal name
#                         'database': db_name,
#                         'name': activity_name,
#                         'product': product_name,
#                         'location': location,  # Keep it for reference
#                         'unit': activity['unit'],
#                         'description': activity.get('comment', None),
#                         'categories': activity.get('categories', None),
#                         'activity type': activity.get('activity type', None),
#                         'production amount': activity.get('production amount', None),
#                         'parameters': activity.get('parameters', None),
#                         'authors': activity.get('authors', None),
#                         'data quality': activity.get('data quality', None)
#                     })
#         else:
#             print(f"⚠️ Database '{db_name}' not found in the current project.")
#
#     return pd.DataFrame(data)