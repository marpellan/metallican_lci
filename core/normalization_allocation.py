import pandas as pd


def normalize_flows(df, production_df, price_df=None, mode='ore',
                    allocation='mass', value_col='value', prod_agg='sum'):
    """
    Normalize LCI flows by ore, metal, or concentrate reference products.

    Adds normalization of:
    - value_min
    - value_mean
    - value_max

    New mode:
    ---------
    - mode='concentrate' : normalize by concentrate streams (columns ending with '_conc')
    """

    df = df.copy()
    prod = production_df.copy()
    df[value_col] = pd.to_numeric(df[value_col], errors='coerce')

    # Normalize helper
    def apply_normalization(out, norm_col):
        for col in ['value', 'value_min', 'value_mean', 'value_max']:
            if col in out.columns:
                out[f"{col}_normalized"] = out[col] / out[norm_col]
        return out

    # Aggregate production per site
    prod = prod.groupby('site_id', as_index=False).agg(prod_agg)

    # Identify metal columns (_t) and concentrate columns (_conc)
    metal_cols = [c for c in prod.columns if c.endswith('_t') and c != 'ore_processed_t']
    conc_cols  = [c for c in prod.columns if c.endswith('_conc')]

    # ===============================================================
    # ---------------------- ORE NORMALIZATION ----------------------
    # ===============================================================
    if mode == 'ore':
        out = df.merge(prod[['site_id', 'ore_processed_t']], on='site_id', how='left')
        out = apply_normalization(out, 'ore_processed_t')
        out['functional_unit'] = 'Ore processed'
        out['allocation_factor'] = 1
        out['normalization_key'] = 'ore'
        return out

    # ===============================================================
    # -------------------- METAL NORMALIZATION ----------------------
    # ===============================================================
    if mode == 'metal':
        melted = prod.melt(id_vars=['site_id'], value_vars=metal_cols,
                           var_name='metal', value_name='mass_t')
        melted['metal'] = melted['metal'].str.replace('_t', '', regex=False)

        melted = melted[melted['mass_t'].notna() & (melted['mass_t'] > 0)]

        # MASS ALLOCATION
        if allocation == 'mass':
            melted['allocation_factor'] = melted.groupby('site_id')['mass_t'] \
                                                .transform(lambda x: x / x.sum())

        # ECONOMIC ALLOCATION
        elif allocation == 'economic':
            if price_df is None:
                raise ValueError("price_df must be provided for economic allocation.")

            price_df = price_df.rename(columns=lambda x: x.strip().lower())
            melted = melted.merge(price_df[['commodity', 'price']],
                                  left_on='metal', right_on='commodity', how='left')
            melted['mass_value'] = melted['mass_t'] * melted['price']
            melted['allocation_factor'] = melted.groupby('site_id')['mass_value'] \
                                                .transform(lambda x: x / x.sum())
        else:
            raise ValueError("allocation must be 'mass' or 'economic'.")

        out = df.merge(
            melted[['site_id', 'metal', 'mass_t', 'allocation_factor']],
            on='site_id', how='inner'
        )

        # Add facility type
        out = out.merge(
            production_df[['site_id', 'facility_type']].drop_duplicates(),
            on='site_id', how='left'
        )

        out = apply_normalization(out, 'mass_t')
        for col in ['value', 'value_min', 'value_mean', 'value_max']:
            if f"{col}_normalized" in out:
                out[f"{col}_normalized"] *= out['allocation_factor']

        out['functional_unit'] = out['metal'] + ', ' + \
            out['facility_type'].str.contains('mining', case=False).map(
                {True: 'usable ore', False: 'metal'})
        out['normalization_key'] = f"metal_{allocation}"
        return out

    # ===============================================================
    # ----------------- CONCENTRATE NORMALIZATION -------------------
    # ===============================================================

    if mode == 'concentrate':
        if len(conc_cols) == 0:
            raise ValueError("No '_conc' columns found in production_df.")

        melted = prod.melt(id_vars=['site_id'], value_vars=conc_cols,
                           var_name='concentrate', value_name='mass_conc')

        melted['concentrate'] = melted['concentrate'].str.replace('_conc', '', regex=False)
        melted = melted[melted['mass_conc'].notna() & (melted['mass_conc'] > 0)]

        if allocation == 'mass':
            melted['allocation_factor'] = melted.groupby('site_id')['mass_conc'] \
                                                .transform(lambda x: x / x.sum())
        elif allocation == 'economic':
            if price_df is None:
                raise ValueError("price_df must be provided for economic allocation.")

            price_df = price_df.rename(columns=lambda x: x.strip().lower())
            melted = melted.merge(price_df[['commodity', 'price']],
                                  left_on='concentrate', right_on='commodity', how='left')

            melted['mass_value'] = melted['mass_conc'] * melted['price']
            melted['allocation_factor'] = melted.groupby('site_id')['mass_value'] \
                                                .transform(lambda x: x / x.sum())
        else:
            raise ValueError("allocation must be 'mass' or 'economic'.")

        out = df.merge(
            melted[['site_id', 'concentrate', 'mass_conc', 'allocation_factor']],
            on='site_id', how='inner'
        )

        out = out.merge(
            production_df[['site_id', 'facility_type']].drop_duplicates(),
            on='site_id', how='left'
        )

        out = apply_normalization(out, 'mass_conc')
        for col in ['value', 'value_min', 'value_mean', 'value_max']:
            if f"{col}_normalized" in out:
                out[f"{col}_normalized"] *= out['allocation_factor']

        out['functional_unit'] = out['concentrate'] + " concentrate"
        out['normalization_key'] = f"concentrate_{allocation}"
        out.loc[out['concentrate'].str.lower() == 'au', 'functional_unit'] = 'Doré'

        # -------------------------------------------------
        # Convert from per tonne to per kg
        # -------------------------------------------------
        for col in ['value', 'value_min', 'value_mean', 'value_max']:
            norm_col = f"{col}_normalized"
            if norm_col in out.columns:
                out[norm_col] = out[norm_col] / 1e3

        out['reference_mass_unit'] = 'kg'


        return out

    # ===============================================================
    # ------------------------- ERROR -------------------------------
    # ===============================================================
    raise ValueError("mode must be one of: 'ore', 'metal', 'concentrate'.")



def normalize_land_flows(
    land_df,
    production_df,
    price_df=None,
    mode="ore",
    allocation="economic",
    lifetime_years=20,
    prod_agg="sum",
):
    """
    Allocate and normalize land use flows for LCA.

    Generates 3 flows per site/product:
      - land transformation (from)
      - land transformation (to)
      - land occupation

    Land formulas:
    ----------------
    Transformation: A / (Q_ref * lifetime_years)
    Occupation:     A / Q_ref

    Passes through site-level metadata unchanged.
    """

    import pandas as pd

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------
    META_COLS = [
        "activity_name",
        "mining_processing_type",
        "archetypes",
        "npv",
        "operation_periods",
        "data_source",
        "value_formula",
        "amount_parameter",
        "parameter_distribution",
    ]

    VALUE_COLS = ["value", "value_min", "value_mean", "value_max"]

    # ------------------------------------------------------------------
    # Copy & numeric safety
    # ------------------------------------------------------------------
    df   = land_df.copy()
    prod = production_df.copy()

    for c in VALUE_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # ------------------------------------------------------------------
    # Aggregate production
    # ------------------------------------------------------------------
    prod = prod.groupby("site_id", as_index=False).agg(prod_agg)

    metal_cols = [c for c in prod.columns if c.endswith("_t") and c != "ore_processed_t"]
    conc_cols  = [c for c in prod.columns if c.endswith("_conc")]

    # ------------------------------------------------------------------
    # Allocation & reference quantity
    # ------------------------------------------------------------------
    if mode == "ore":
        out = df.merge(
            prod[["site_id", "ore_processed_t"]],
            on="site_id",
            how="left"
        )
        out["Q_ref"] = out["ore_processed_t"]
        out["allocation_factor"] = 1.0
        out["functional_unit"] = "Ore processed"
        out["normalization_key"] = "land_ore"

    elif mode == "metal":
        if allocation != "economic":
            raise ValueError("Land at metal level must be economically allocated.")

        if price_df is None:
            raise ValueError("price_df required for economic allocation.")

        melted = prod.melt(
            id_vars=["site_id"],
            value_vars=metal_cols,
            var_name="metal",
            value_name="mass_t",
        )
        melted["metal"] = melted["metal"].str.replace("_t", "", regex=False)
        melted = melted[melted["mass_t"] > 0]

        price_df = price_df.rename(columns=str.lower)
        melted = melted.merge(
            price_df[["commodity", "price"]],
            left_on="metal",
            right_on="commodity",
            how="left"
        )

        melted["econ_value"] = melted["mass_t"] * melted["price"]
        melted["allocation_factor"] = (
            melted.groupby("site_id")["econ_value"]
            .transform(lambda x: x / x.sum())
        )

        out = df.merge(
            melted[["site_id", "metal", "mass_t", "allocation_factor"]],
            on="site_id",
            how="inner"
        )

        out["Q_ref"] = out["mass_t"]
        out["functional_unit"] = out["metal"] + ", metal"
        out["normalization_key"] = "land_metal_economic"

    elif mode == "concentrate":
        if allocation != "economic":
            raise ValueError("Land at concentrate level must be economically allocated.")

        if price_df is None:
            raise ValueError("price_df required for economic allocation.")

        melted = prod.melt(
            id_vars=["site_id"],
            value_vars=conc_cols,
            var_name="concentrate",
            value_name="mass_conc",
        )
        melted["concentrate"] = (
            melted["concentrate"].str.replace("_conc", "", regex=False)
        )
        melted = melted[melted["mass_conc"] > 0]

        price_df = price_df.rename(columns=str.lower)
        melted = melted.merge(
            price_df[["commodity", "price"]],
            left_on="concentrate",
            right_on="commodity",
            how="left"
        )

        melted["econ_value"] = melted["mass_conc"] * melted["price"]
        melted["allocation_factor"] = (
            melted.groupby("site_id")["econ_value"]
            .transform(lambda x: x / x.sum())
        )

        out = df.merge(
            melted[["site_id", "concentrate", "mass_conc", "allocation_factor"]],
            on="site_id",
            how="inner"
        )

        out["Q_ref"] = out["mass_conc"]
        out["functional_unit"] = out["concentrate"] + " concentrate"
        out["normalization_key"] = "land_concentrate_economic"

    else:
        raise ValueError("mode must be 'ore', 'metal', or 'concentrate'.")

    # Detect which metadata columns are present
    meta_cols_present = [c for c in META_COLS if c in out.columns]

    # ------------------------------------------------------------------
    # Build land LCI records
    # ------------------------------------------------------------------
    records = []

    for _, r in out.iterrows():
        for flow in ["transformation_from", "transformation_to", "occupation"]:

            rec = {
                "site_id": r["site_id"],
                "flow_type": flow,
                "functional_unit": r["functional_unit"],
                "normalization_key": r["normalization_key"],
                "allocation_factor": r["allocation_factor"],
                "unit": "m2" if flow != "occupation" else "m2*year",
            }

            # pass-through metadata
            for c in meta_cols_present:
                rec[c] = r[c]

            # land equations
            for c in VALUE_COLS:
                A_alloc = r[c] * r["allocation_factor"]

                if flow == "occupation":
                    rec[c + "_normalized"] = A_alloc / r["Q_ref"]
                else:
                    rec[c + "_normalized"] = A_alloc / (
                        r["Q_ref"] * lifetime_years
                    )

            records.append(rec)

    out_df = pd.DataFrame(records)

    # -------------------------------------------------
    # Convert from per tonne to per kg
    # -------------------------------------------------
    for col in ["value", "value_min", "value_mean", "value_max"]:
        norm_col = f"{col}_normalized"
        if norm_col in out_df.columns:
            out_df[norm_col] = out_df[norm_col] / 1e3

    out_df["reference_mass_unit"] = "kg"

    return out_df
