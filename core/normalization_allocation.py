import pandas as pd


def normalize_flows(df, production_df, price_df=None, mode='ore',
                    allocation='mass', value_col='value', prod_agg='sum'):
    """
    Normalize LCI flows by ore, metal, or concentrate reference products.

    New mode:
    ---------
    - mode='concentrate' : normalize by concentrate streams (columns ending with '_conc')

    Other modes unchanged.
    """

    df = df.copy()
    prod = production_df.copy()
    df[value_col] = pd.to_numeric(df[value_col], errors='coerce')

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
        out['value_normalized'] = out[value_col] / out['ore_processed_t']
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

        out['value_normalized'] = (out[value_col] / out['mass_t']) * out['allocation_factor']
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

        # melt concentrate columns
        melted = prod.melt(id_vars=['site_id'], value_vars=conc_cols,
                           var_name='concentrate', value_name='mass_conc')

        # clean names: "Cu_conc" -> "Cu"
        melted['concentrate'] = melted['concentrate'].str.replace('_conc', '', regex=False)

        # filter zero entries
        melted = melted[melted['mass_conc'].notna() & (melted['mass_conc'] > 0)]

        # MASS ALLOCATION (default)
        if allocation == 'mass':
            melted['allocation_factor'] = melted.groupby('site_id')['mass_conc'] \
                                                .transform(lambda x: x / x.sum())

        # ECONOMIC ALLOCATION (if concentrate prices exist)
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

        # merge flows
        out = df.merge(
            melted[['site_id', 'concentrate', 'mass_conc', 'allocation_factor']],
            on='site_id', how='inner'
        )

        # Add facility type
        out = out.merge(
            production_df[['site_id', 'facility_type']].drop_duplicates(),
            on='site_id', how='left'
        )

        out['value_normalized'] = (out[value_col] / out['mass_conc']) * out['allocation_factor']
        out['functional_unit'] = out['concentrate'] + " concentrate"
        out['normalization_key'] = f"concentrate_{allocation}"

        return out

    # ===============================================================
    # ------------------------- ERROR -------------------------------
    # ===============================================================
    raise ValueError("mode must be one of: 'ore', 'metal', 'concentrate'.")