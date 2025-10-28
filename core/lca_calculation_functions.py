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


### LCI ###
def search_activity(database_name, activity_name, ref_product, location):
    """
    Function to find a specific activity based on its name and reference product in a BW database
    """
    db = bw.Database(database_name)
    matches = [ds for ds in db if ds["name"] == activity_name
               and ds["reference product"] == ref_product
               and ds["location"] == location]

    if matches:
        print(f"Match found in {database_name}:")
        for match in matches:
            print(match)
    else:
        print(
            f"No match found in {database_name} for activity '{activity_name}', product '{ref_product}', location '{location}'")


def filter_ecoinvent_activities(databases_to_include, products_to_include, locations_to_include=None):
    """
    Extracts activities from Ecoinvent databases based on product and location filters.

    Parameters:
    - databases_to_include (list): List of Ecoinvent databases to search.
    - products_to_include (list): List of product/activity keywords to match.
    - locations_to_include (list, optional): List of locations to filter (default: None, includes all locations).

    Returns:
    - pd.DataFrame: DataFrame with filtered Ecoinvent activities.
    """
    data = []

    for db_name in databases_to_include:
        if db_name in bw.databases:  # Check if the database exists
            db = bw.Database(db_name)
            for activity in db:
                product_name = activity.get('reference product', None)
                activity_name = activity['name']
                location = activity.get('location', None)

                # Match product keyword in either 'reference product' or 'activity name'
                matched_metal = next(
                    (p for p in products_to_include if
                     (product_name and re.search(rf'\b{p}\b', product_name, re.IGNORECASE)) or
                     (activity_name and re.search(rf'\b{p}\b', activity_name, re.IGNORECASE))
                    ),
                    None
                )

                # Apply location filter if specified
                if matched_metal and (locations_to_include is None or location in locations_to_include):
                    data.append({
                        'metal': matched_metal,  # Add the identified metal name
                        'database': db_name,
                        'name': activity_name,
                        'product': product_name,
                        'location': location,  # Keep it for reference
                        'unit': activity['unit'],
                        'description': activity.get('comment', None),
                        'categories': activity.get('categories', None),
                        'activity type': activity.get('activity type', None),
                        'production amount': activity.get('production amount', None),
                        'parameters': activity.get('parameters', None),
                        'authors': activity.get('authors', None),
                        'data quality': activity.get('data quality', None)
                    })
        else:
            print(f"⚠️ Database '{db_name}' not found in the current project.")

    return pd.DataFrame(data)


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


def compute_midpoint_contributions(
    inventories,
    amount=1,
    damage_version="IMPACT World+ Damage 2.1_regionalized for ecoinvent v3.10"
):
    """
    For each activity in `inventories`, computes midpoint→endpoint damage shares
    for IW+ 2.1 Human health and Ecosystem quality.

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


def first_tier_contributions(activity, commodity_label, method_id,
                                     amount=1, threshold=0.01):
    """
    Builds a DataFrame of direct (first‑tier) technosphere contributions
    for one commodity, filtered by a percentage cutoff of total endpoint impact.

    Parameters
    ----------
    activity : brightway2 Activity
        The target activity.
    commodity_label : str
        Name of the commodity (for the 'Commodity' column).
    method_id : tuple or str
        The LCIA method identifier (e.g., a triple for IW+ endpoint).
    amount : float
        Functional unit for the target activity.
    threshold : float (0-1)
        Fraction of total impact; only include providers with share >= threshold.

    Returns
    -------
    pd.DataFrame with columns:
      - Commodity     (str)
      - Activity      (provider name)
      - Product       (provider reference product)
      - Location      (provider location)
      - Impact_score  (absolute contribution to endpoint)
      - Share_%       (percentage of total endpoint impact)
    """
    # 1) compute total endpoint impact of the main activity
    lca_main = bw.LCA({activity.key: amount})
    lca_main.lci()
    lca_main.switch_method(method_id)
    lca_main.lcia()
    total_score = lca_main.score or 1e-30

    rows = []
    # 2) loop direct technosphere inputs
    for exc in activity.technosphere():
        provider = exc.input
        unit_amt = getattr(exc, "amount", exc.get("amount", 1))

        # 3) pull provider details
        name     = provider.get("name", "<unknown>")
        product  = provider.get("reference product", "<unknown>")
        location = provider.get("location", "<unknown>")

        # 4) run LCA on provider to get its unit CF
        lca_p = bw.LCA({provider.key: 1})
        lca_p.lci()
        lca_p.switch_method(method_id)
        lca_p.lcia()
        cf_unit = lca_p.score

        # 5) compute contribution & share
        contr = cf_unit * unit_amt * amount
        share_frac = contr / total_score  # fraction of total
        share_pct = share_frac * 100      # percent

        if share_frac >= threshold:
            rows.append({
                "Commodity":     commodity_label,
                "Activity":      name,
                "Product":       product,
                "Location":      location,
                "Impact_score":  contr,
                "Share_%":       share_pct
            })

    # 6) assemble and sort
    df = pd.DataFrame(rows)
    return df.sort_values("Share_%", ascending=False).reset_index(drop=True)



def first_tier_contributions_batch(inventory_dict, method_id, amount=1, threshold=0.01):
    """
    Runs first-tier contribution analysis for multiple commodities at once.

    Parameters
    ----------
    inventory_dict : dict
        Mapping of commodity name → Brightway Activity
        Example: {'Copper concentrate': activity1, 'Nickel concentrate': activity2}
    method_id : tuple or str
        LCIA method identifier (e.g., ('ReCiPe Endpoint (H,A)', 'total', 'human health'))
    amount : float
        Functional unit for each activity.
    threshold : float
        Minimum share (fraction of total) to include in output.

    Returns
    -------
    pd.DataFrame
        Combined results with columns:
        ['Commodity', 'Activity', 'Product', 'Location', 'Impact_score', 'Share_%']
    """
    all_results = []

    for commodity, activity in inventory_dict.items():
        print(f"🔹 Processing {commodity} ...")

        # Compute total impact for this activity
        lca_main = bw.LCA({activity.key: amount})
        lca_main.lci()
        lca_main.switch_method(method_id)
        lca_main.lcia()
        total_score = lca_main.score or 1e-30

        rows = []
        # Loop over direct technosphere inputs
        for exc in activity.technosphere():
            provider = exc.input
            unit_amt = getattr(exc, "amount", exc.get("amount", 1))

            # Provider details
            name     = provider.get("name", "<unknown>")
            product  = provider.get("reference product", "<unknown>")
            location = provider.get("location", "<unknown>")

            # Provider's unit impact
            lca_p = bw.LCA({provider.key: 1})
            lca_p.lci()
            lca_p.switch_method(method_id)
            lca_p.lcia()
            cf_unit = lca_p.score

            # Contribution and share
            contr = cf_unit * unit_amt * amount
            share_frac = contr / total_score
            share_pct = share_frac * 100

            if share_frac >= threshold:
                rows.append({
                    "Commodity":     commodity,
                    "Activity":      name,
                    "Product":       product,
                    "Location":      location,
                    "Impact_score":  contr,
                    "Share_%":       share_pct
                })

        df = pd.DataFrame(rows)
        if not df.empty:
            all_results.append(df.sort_values("Share_%", ascending=False))

    if all_results:
        return pd.concat(all_results, ignore_index=True)
    else:
        return pd.DataFrame(columns=["Commodity", "Activity", "Product", "Location", "Impact_score", "Share_%"])


def process_contributions(activity, commodity_label, method_id,
                                     amount=1, threshold=0.05):
    """
    Full‑chain process contributions using Brightway’s ContributionAnalysis,
    filtered by a percentage cutoff of total endpoint impact.

    Parameters
    ----------
    activity : brightway2 Activity
        The target activity.
    commodity_label : str
        Name of the commodity (for the 'Commodity' column).
    method_id : tuple or str
        The LCIA method identifier (e.g. the IW+ endpoint triple).
    amount : float
        Functional unit for the target activity.
    threshold : float (0–1)
        Only include processes whose share ≥ threshold fraction of total impact.

    Returns
    -------
    pd.DataFrame with columns:
      - Commodity     (str)
      - Activity      (provider name)
      - Product       (provider reference product)
      - Location      (provider location)
      - Impact_score  (absolute contribution to endpoint)
      - Share_%       (percentage of total endpoint impact)
    """
    # 1) Run and characterize the LCA on the target
    lca = bw.LCA({activity.key: amount})
    lca.lci()
    lca.switch_method(method_id)
    lca.lcia()
    total = lca.score or 1e-30

    # 2) Use Brightway ContributionAnalysis
    ca = ba.ContributionAnalysis()
    # annotated_top_processes returns (score, supply, process_key)
    raw = ca.annotated_top_processes(
        lca,
        names=False,
        limit=threshold,
        limit_type='percent'
    )

    # 3) Build DataFrame rows
    rows = []
    for score, supply, proc_key in raw:
        act = bw.get_activity(proc_key)
        rows.append({
            "Commodity":     commodity_label,
            "Activity":      act.get("name", "<unknown>"),
            "Product":       act.get("reference product", "<unknown>"),
            "Location":      act.get("location", "<unknown>"),
            "Impact_score":  float(score),
            "Share_%":       float(score / total * 100)
        })

    # 4) Assemble and sort
    df = pd.DataFrame(rows)
    return df.sort_values("Share_%", ascending=False).reset_index(drop=True)


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


def create_pedigree_matrix(pedigree_scores: tuple, exc_amount: float):
    """
    Function from Istrate et al (2024)

    This function returns a dict containing the pedigree matrix dict and loc and scale values
    that can be used to update exchanges in a dataset dict

    The pedigree matrix dictionary is created using the scores provided in the LCI Excel file.

    The code to calcualte the loc and scale values is based on https://github.com/brightway-lca/pedigree_matrix,
    which is published by Chris Mutel under an BSD 3-Clause License (2021).

    :param pedigree_scores: tuple of pedigree scores
    :param exc_amount: exchange amount
    :return dict:
    """

    VERSION_2 = {
        "reliability": (1.0, 1.54, 1.61, 1.69, 1.69),
        "completeness": (1.0, 1.03, 1.04, 1.08, 1.08),
        "temporal correlation": (1.0, 1.03, 1.1, 1.19, 1.29),
        "geographical correlation": (1.0, 1.04, 1.08, 1.11, 1.11),
        "further technological correlation": (1.0, 1.18, 1.65, 2.08, 2.8),
        "sample size": (1.0, 1.0, 1.0, 1.0, 1.0),
    }

    pedigree_scores_dict = {
        'reliability': pedigree_scores[0],
        'completeness': pedigree_scores[1],
        'temporal correlation': pedigree_scores[2],
        'geographical correlation': pedigree_scores[3],
        'further technological correlation': pedigree_scores[4]
    }

    assert len(pedigree_scores) in (5, 6), "Must provide either 5 or 6 factors"
    if len(pedigree_scores) == 5:
        pedigree_scores = pedigree_scores + (1,)

    factors = [VERSION_2[key][index - 1] for key, index in pedigree_scores_dict.items()]

    basic_uncertainty: float = 1.0
    values = [basic_uncertainty] + factors

    scale = math.sqrt(sum([math.log(x) ** 2 for x in values])) / 2
    loc = math.log(abs(exc_amount))

    pedigree_dict = {
        'uncertainty type': 2,
        'loc': loc,
        'scale': scale,
        "pedigree": pedigree_scores_dict,
    }
    return pedigree_dict


### Scale up ###
# Calculate projected impacts using the mapping
# def calculate_projected_impacts(production_df, impact_df, mapping):
#     projections = []
#
#     for mineral in production_df['Commodity'].unique():
#         # Use the mapping dictionary to get the corresponding raw material
#         raw_material = mapping.get(mineral)
#
#         if raw_material:
#             # Fetch impact factors for the mapped raw material
#             material_impacts = impact_df[impact_df['Commodity'] == raw_material]
#
#             if not material_impacts.empty:
#                 impacts_per_kt = material_impacts.iloc[0, 1:].to_dict()  # Extract impact per kilotonne as a dict
#
#                 # Filter production data for the mineral
#                 mineral_data = production_df[production_df['Commodity'] == mineral]
#
#                 for _, row in mineral_data.iterrows():
#                     year = row['Year']
#                     production_kilotons = row['Value']
#
#                     # Calculate impacts for each category
#                     annual_impacts = {f"{category}": production_kilotons * impact_per_kt * 1000000
#                                       for category, impact_per_kt in impacts_per_kt.items()}
#                     annual_impacts['Year'] = year
#                     annual_impacts['Commodity'] = mineral
#                     projections.append(annual_impacts)
#
#     # Convert list of dictionaries to DataFrame
#     projected_impacts_df = pd.DataFrame(projections)
#
#     # Reorder columns to have 'Year' and 'Commodity' first
#     impact_columns = [col for col in projected_impacts_df.columns if col not in ['Year', 'Commodity']]
#     projected_impacts_df = projected_impacts_df[['Year', 'Commodity'] + impact_columns]
#
#     return projected_impacts_df