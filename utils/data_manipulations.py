import pandas as pd


def merge_main_and_group(df, main_df,
                       main_key='main_id', group_key='facility_group_id',
                       cols_to_add=['facility_name', 'facility_group_name', 'mining_processing_type', 'commodities'], fill_from_group=True, verbose=True):
    """
    Robustly add columns from main_df to df by mapping values using dictionaries.
    Preserves row order and does not drop duplicates.
    """
    import pandas as pd

    df_out = df.copy()
    if cols_to_add is None:
        candidate = ['facility_name', 'facility_group_name', 'mining_processing_type', 'commodities']
        cols_to_add = [c for c in candidate if c in main_df.columns]
    else:
        cols_to_add = [c for c in cols_to_add if c in main_df.columns]

    # Normalize keys (string, stripped)
    df_out['_k_main'] = df_out[main_key].astype('string').str.strip().replace({'nan': None})
    df_out['_k_group'] = df_out[group_key].astype('string').str.strip().replace({'nan': None}) if group_key in df_out.columns else pd.Series([None]*len(df_out), index=df_out.index)

    # Prepare main_table keys
    main_df = main_df.copy()
    if main_key in main_df.columns:
        main_df['_k_main'] = main_df[main_key].astype('string').str.strip().replace({'nan': None})
    if group_key in main_df.columns:
        main_df['_k_group'] = main_df[group_key].astype('string').str.strip().replace({'nan': None})

    diagnostics = {'main_matches': {}, 'group_matches': {}, 'final_nonnull': {}}

    for col in cols_to_add:
        # mapping from main_id
        map_main = {}
        if main_key in main_df.columns:
            map_main = main_df.dropna(subset=['_k_main']).set_index('_k_main')[col].to_dict()
        mapped_main = df_out['_k_main'].map(map_main)

        # mapping from group_id
        mapped_group = pd.Series(pd.NA, index=df_out.index)
        if fill_from_group and (group_key in main_df.columns):
            map_group = main_df.dropna(subset=['_k_group']).set_index('_k_group')[col].to_dict()
            mapped_group = df_out['_k_group'].map(map_group)

        # Combine: prefer main mapping, else group mapping
        combined = mapped_main.where(~mapped_main.isna(), mapped_group)
        df_out[col] = combined

        diagnostics['main_matches'][col] = int(mapped_main.notna().sum())
        diagnostics['group_matches'][col] = int(mapped_group.notna().sum())
        diagnostics['final_nonnull'][col] = int(df_out[col].notna().sum())

    # Clean up temp columns and ensure new columns are on the right
    df_out.drop(columns=['_k_main', '_k_group'], inplace=True, errors='ignore')
    original_cols = list(df.columns)
    new_cols = [c for c in cols_to_add if c not in original_cols]
    df_out = df_out[original_cols + new_cols]

    # if verbose:
    #     print("Diagnostics for mapping (v2):")
    #     for col in cols_to_add:
    #         print(f"  {col}: main matches={diagnostics['main_matches'][col]}, group matches={diagnostics['group_matches'][col]}, final non-null={diagnostics['final_nonnull'][col]}")

    return df_out


# def merge_without_suffixes(
#     left: pd.DataFrame,
#     right: pd.DataFrame,
#     keys: Tuple[str, str] = ("main_id", "facility_group_id"),
#     how: str = "left",
#     right_prefix: str = ""  # leave empty to add only non-overlapping cols
# ) -> pd.DataFrame:
#     """
#     Merge two DataFrames on keys, adding only *new* columns from right to avoid _x/_y.
#     - Ensures right has unique keys to prevent row multiplication.
#     - Drops overlapping non-key columns from right before merge.
#     - Returns a clean merged DataFrame with no suffixes.
#     """
#     key_cols = list(keys)
#
#     # Ensure key types align
#     for k in key_cols:
#         if k in left.columns and k in right.columns:
#             # Cast to string to be robust to mixed types; change to category/int if you prefer
#             left[k] = left[k].astype(str)
#             right[k] = right[k].astype(str)
#
#     # Deduplicate right on keys (keep first occurrence)
#     if not right.duplicated(subset=key_cols, keep=False).any():
#         right_dedup = right.copy()
#     else:
#         # If duplicates exist, keep the first row per keys (you can change strategy if needed)
#         right_dedup = right.drop_duplicates(subset=key_cols, keep="first")
#
#     # Determine overlapping non-key columns and drop them from right
#     overlap = [c for c in right_dedup.columns if c in left.columns and c not in key_cols]
#     right_cols_to_use = [c for c in right_dedup.columns if c not in overlap]
#
#     # Optionally add a prefix to new columns from right (disabled by default to keep names clean)
#     if right_prefix:
#         rename_map = {
#             c: f"{right_prefix}{c}" for c in right_cols_to_use if c not in key_cols
#         }
#         right_dedup = right_dedup[right_cols_to_use].rename(columns=rename_map)
#     else:
#         right_dedup = right_dedup[right_cols_to_use]
#
#     # Perform merge
#     merged = left.merge(right_dedup, on=key_cols, how=how, validate=None)
#
#     return merged

def prepare_normalization_data(df):
    '''
    Fonction pour nettoyer le DataFrame en vue de la normalisation.
    Règles :
    1) Pour chaque (main_id, facility_group_id), garder uniquement les lignes avec data_type = 'Production'.
    2) Pour chaque (main_id, facility_group_id), garder la ligne avec le reference_point le plus prioritaire.
       Si plusieurs commodities, les agréger (somme des valeurs, concaténation des commodities).
    3) Ordre de priorité : 'Crude ore' > 'Total extraction' > 'Intermediate metal produced' > 'Refined metal produced' > 'Usable ore'.
    Return :
    cleaned_df : DataFrame nettoyé avec les mêmes colonnes.
    '''
    # 1. Filtrer pour ne garder que les lignes avec data_type = 'Production'
    df_filtered = df[df['data_type'] == 'Production'].copy()

    # 2. Définition des priorités (plus le chiffre est bas, plus la priorité est haute)
    priority_order = {
        'Crude ore': 1,
        'Total extraction': 2,
        'Intermediate metal produced': 3,
        'Refined metal produced': 4,
        'Usable ore': 5
    }

    # 3. Préparation des clés de groupe : remplir NaN pour main_id et facility_group_id
    df_filtered['main_id'] = df_filtered['main_id'].fillna('NA_ID')
    df_filtered['facility_group_id'] = df_filtered['facility_group_id'].fillna('NA_ID')
    id_cols = ['main_id', 'facility_group_id', 'year', 'geography']

    # 4. Créer la colonne de priorité
    df_filtered['priority'] = df_filtered['reference_point'].map(priority_order).fillna(99)

    # 5. Trier le DataFrame par priorité pour identifier le meilleur niveau
    df_sorted = df_filtered.sort_values(by=id_cols + ['priority'], ascending=[True] * len(id_cols) + [True])

    # 6. Identification du Meilleur Niveau de Priorité (pour tout le groupe)
    best_priority_levels = df_sorted.groupby(id_cols)['priority'].min().reset_index().rename(
        columns={'priority': 'best_priority_level'}
    )

    # 7. Filtrage pour l'agrégation
    df_merged = df_sorted.merge(best_priority_levels, on=id_cols, how='left')
    df_to_sum = df_merged[df_merged['priority'] == df_merged['best_priority_level']].copy()

    # 8. Assurer que les colonnes sont numériques pour la somme
    df_to_sum['value'] = pd.to_numeric(df_to_sum['value'], errors='coerce')
    df_to_sum['value_tonnes'] = pd.to_numeric(df_to_sum['value_tonnes'], errors='coerce')

    # 9. Agrégation (Somme des valeurs et concaténation des commodities)
    aggregated_data = df_to_sum.groupby(id_cols).agg(
        value_sum=('value', 'sum'),
        value_tonnes_sum=('value_tonnes', 'sum'),
        commodity_agg=('commodity', lambda x: ', '.join(sorted(x.unique())))
    ).reset_index()

    # 10. Extraction des Métadonnées et Fusion
    metadata_cols = [col for col in df_sorted.columns if col not in ['value', 'value_tonnes', 'priority']]
    cleaned_df = df_sorted.drop_duplicates(subset=id_cols, keep='first')[metadata_cols]

    # 11. Fusionner les métadonnées avec les valeurs agrégées
    cleaned_df = cleaned_df.merge(aggregated_data, on=id_cols, how='left')

    # 12. Remplacer les colonnes de valeur et de commodity avec les valeurs agrégées
    cleaned_df['value'] = cleaned_df['value_sum']
    cleaned_df['value_tonnes'] = cleaned_df['value_tonnes_sum']
    cleaned_df['commodity'] = cleaned_df['commodity_agg']

    # 13. Nettoyage Final et Restauration
    cleaned_df['main_id'] = cleaned_df['main_id'].replace('NA_ID', None)
    cleaned_df['facility_group_id'] = cleaned_df['facility_group_id'].replace('NA_ID', None)

    # 14. Rétablir l'ordre original des colonnes du DF d'entrée
    original_cols = [col for col in df.columns if col in cleaned_df.columns]
    cols_to_drop = ['best_priority_level', 'value_sum', 'value_tonnes_sum', 'commodity_agg', 'priority']
    cleaned_df = cleaned_df.drop(columns=[col for col in cols_to_drop if col in cleaned_df.columns])

    return cleaned_df[original_cols]


def normalize_by_production(df, production_df, value_col='value', prod_col='value_tonnes', prod_agg='sum'):

    df = df.copy()
    # ensure numeric
    df[value_col] = pd.to_numeric(df[value_col], errors='coerce')
    prod = production_df.copy()
    prod[prod_col] = pd.to_numeric(prod[prod_col], errors='coerce')

    # aggregate to unique per key
    main_prod = (prod.dropna(subset=['main_id'])
                    .groupby('main_id', as_index=False)[prod_col]
                    .agg(prod_agg)
                    .rename(columns={prod_col: 'value_tonnes_main'}))
    group_prod = (prod.dropna(subset=['facility_group_id'])
                     .groupby('facility_group_id', as_index=False)[prod_col]
                     .agg(prod_agg)
                     .rename(columns={prod_col: 'value_tonnes_group'}))

    # safe 1:1 merges
    out = df.merge(main_prod, on='main_id', how='left').merge(group_prod, on='facility_group_id', how='left')

    # prefer main_id match, fallback to facility_group_id
    out['value_tonnes_match'] = out['value_tonnes_main'].combine_first(out['value_tonnes_group'])
    out['value_normalized'] = out[value_col] / out['value_tonnes_match']

    # diagnostics
    out['normalization_key'] = None
    out.loc[out['value_tonnes_main'].notna(), 'normalization_key'] = 'main_id'
    out.loc[out['value_tonnes_main'].isna() & out['value_tonnes_group'].notna(), 'normalization_key'] = 'facility_group_id'
    return out


def get_info_for_ids(df, id_pairs):
    # Extract facility_group_ids from id_pairs
    facility_group_ids = {id_pair[1] for id_pair in id_pairs if pd.notna(id_pair[1])}

    # Create a boolean mask for rows that match any of the id pairs or have a facility_group_id in the set
    mask = df.apply(lambda row: (
        (row['main_id'], row['facility_group_id']) in id_pairs or
        row['facility_group_id'] in facility_group_ids
    ), axis=1)

    # Use the mask to filter the DataFrame
    filtered_df = df[mask]
    return filtered_df


def get_production_data(production_df, reference_priority=None):
    """
    Filters the production DataFrame to keep only the highest priority reference_point per group.

    Args:
        production_df (pd.DataFrame): Input DataFrame containing production data.
        reference_priority (dict): Dictionary mapping reference_point to priority.
                                   Defaults to {"Crude ore": 0, "Total extraction": 1}.

    Returns:
        pd.DataFrame: Filtered DataFrame with only the highest priority reference_point per group.
    """
    if reference_priority is None:
        reference_priority = {"Crude ore": 0, "Total extraction": 1}

    # Keep only relevant rows
    df = production_df[
        (production_df['prod_id'].str.startswith('PROD')) &
        (production_df['reference_point'].isin(reference_priority.keys()))
    ].copy()

    # Assign priority score
    df["priority"] = df["reference_point"].map(reference_priority)

    # Sort and keep only the best reference_point per group
    df = df.sort_values(["main_id", "facility_group_id", "priority"])
    result = df.drop_duplicates(subset=["main_id", "facility_group_id"], keep="first").drop(columns="priority")

    return result


# # Add the MDO_URL column to the archetypes_table with a merge from the main_id and facility_group_id columns
# def add_mdo_url(archetypes_df, main_df):
#     merged_df = pd.merge(
#         archetypes_df,
#         main_df[['main_id', 'facility_group_id', 'MDO_URL']],
#         on=['main_id', 'facility_group_id'],
#         how='left'
#     )
#     return merged_df


def aggregate_biosphere_facility_groups(df, remove_individuals=False):
    """
    Aggregate environmental flows by facility_group_id when multiple facilities exist.
    Creates new rows (no main_id, no facility_name) with summed values and
    aggregated lists of commodities and mining_processing_type.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing environmental flows.
    remove_individuals : bool, default=False
        If True, removes individual facilities that belong to a facility group
        after aggregation (to avoid double counting).

    Returns
    -------
    pd.DataFrame
        DataFrame with aggregated facility-group rows appended (or replacing individuals).
    """
    df = df.copy()

    def concat_unique(series):
        """Concatenate unique, non-null, case-insensitive values."""
        vals = series.dropna().astype(str).unique()
        # Split comma-separated entries, normalize case
        split_vals = {v.strip().capitalize() for val in vals for v in val.split(',')}
        return ', '.join(sorted(split_vals)) if split_vals else None

    # Aggregate within each facility group
    grouped = (
        df[df["facility_group_id"].notna()]
        .groupby(
            [
                "facility_group_id",
                "year",
                "compartment_name",
                "substance_id",
                "flow_direction",
                "release_pathway",
                "unit",
                "source_id",
            ],
            dropna=False
        )
        .agg({
            "value": "sum",
            #"facility_group_name": "first",
            "company_id": "first"
            #"commodities": concat_unique,
            #"mining_processing_type": concat_unique
        })
        .reset_index()
    )

    # Build aggregated rows
    grouped["main_id"] = None
    grouped["facility_name"] = ""  # leave blank
    grouped["comment"] = "Aggregated value from multiple facilities"

    # Reorder columns to align with original DataFrame
    aggregated_df = grouped.reindex(columns=df.columns.intersection(grouped.columns), fill_value=None)

    # Optionally remove individual facilities
    if remove_individuals:
        df = df[~df["facility_group_id"].isin(grouped["facility_group_id"].unique())]

    # Combine original and aggregated
    combined_df = pd.concat([df, aggregated_df], ignore_index=True)
    return combined_df


def add_site_id(
    df: pd.DataFrame,
    main_col: str = "main_id",
    group_col: str = "facility_group_id",
    out_col: str = "site_id",
) -> pd.DataFrame:
    """
    Create a single canonical site_id column:
    - Prefer `main_id` when present, otherwise `facility_group_id`
    - Normalize to uppercase, strip whitespace
    """
    # copy to avoid mutating caller
    df = df.copy()

    # make sure both columns exist even if missing
    if main_col not in df.columns:
        df[main_col] = pd.NA
    if group_col not in df.columns:
        df[group_col] = pd.NA

    # unify true missing values
    df[main_col] = df[main_col].replace({None: pd.NA, "": pd.NA, "nan": pd.NA})
    df[group_col] = df[group_col].replace({None: pd.NA, "": pd.NA, "nan": pd.NA})

    # prefer main_id, fallback to facility_group_id
    site = df[main_col].fillna(df[group_col])

    # normalize: string, strip, uppercase
    site = site.astype(str).str.strip()
    site = site.mask(site.eq("nan"))  # undo string "nan"
    site = site.fillna(pd.NA)
    #site = site.str.upper()

    df[out_col] = site
    return df

def build_activity_name(row, df):
    """
    Build an LCI activity name from mining type, commodities, and facility name.
    The function scans for all *_t columns (except 'ore_processed_t') in df.
    """
    # ---- 1️⃣ Determine operation type ----
    mpt = str(row.get("mining_processing_type", "")).lower()
    parts = []

    if "open-pit" in mpt and "underground" in mpt:
        parts.append("Open-pit and underground mining")
    elif "open-pit" in mpt:
        parts.append("Open-pit mining")
    elif "underground" in mpt:
        parts.append("Underground mining")

    if "concentrator" in mpt:
        parts.append("and beneficiation")

    op = " ".join(parts)

    # ---- 2️⃣ Determine commodities ----
    commodities = [
        col.replace("_t", "")
        for col in df.columns
        if col.endswith("_t") and col != "ore_processed_t" and pd.notna(row.get(col)) and row[col] != 0
    ]
    commodities_str = " and ".join(commodities)

    # ---- 3️⃣ Choose facility name ----
    facility = (
        row.get("facility_name")
        if pd.notna(row.get("facility_name"))
        else row.get("facility_group_name", "")
    )

    # ---- 4️⃣ Build name ----
    if commodities_str and op and facility:
        return f"{commodities_str}, {op} at {facility}"
    elif commodities_str and op:
        return f"{commodities_str}, {op}"
    elif commodities_str and facility:
        return f"{commodities_str} at {facility}"
    elif commodities_str:
        return commodities_str
    else:
        return None