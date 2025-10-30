import brightway2 as bw
import pandas as pd
from core.constants import CA_provinces


def normalize_flows(df, production_df, price_df=None, mode='ore',
                    allocation='mass', value_col='value', prod_agg='sum'):
    """
    Normalize LCI flows by production reference (ore processed or final metal output).

    Parameters
    ----------
    df : pd.DataFrame
        Flow dataframe containing at least 'main_id' and a numeric column (value_col).
    production_df : pd.DataFrame
        Production dataframe with columns like ['main_id', 'ore_processed_t', 'Au_t', 'Ag_t', ...].
    price_df : pd.DataFrame, optional
        Must contain ['commodity', 'price'] if allocation='economic'.
    mode : {'ore', 'metal'}
        - 'ore' divides by ore_processed_t.
        - 'metal' divides by each metal_t and allocates among metals.
    allocation : {'mass', 'economic'}
        Allocation key when mode='metal'.
    value_col : str, default 'value'
        Column in df to normalize.
    prod_agg : str, default 'sum'
        Aggregation for production_df if multiple rows per main_id.

    Returns
    -------
    pd.DataFrame
        Normalized dataframe with:
        - value_normalized
        - Reference product
        - allocation_factor (if mode='metal')
        - normalization_key ('ore' or 'metal_mass' / 'metal_economic')
    """

    df = df.copy()
    prod = production_df.copy()
    df[value_col] = pd.to_numeric(df[value_col], errors='coerce')
    prod = prod.groupby('main_id', as_index=False).agg(prod_agg)

    # Identify metal columns (ending with _t except ore_processed_t)
    metal_cols = [c for c in prod.columns if c.endswith('_t') and c != 'ore_processed_t']

    if mode == 'ore':
        out = df.merge(prod[['main_id', 'ore_processed_t']], on='main_id', how='left')
        out['value_normalized'] = out[value_col] / out['ore_processed_t']
        out['functional_unit'] = 'Ore processed'
        out['allocation_factor'] = 1
        out['normalization_key'] = 'ore'
        return out

    # ---- METAL-BASED NORMALIZATION ----
    melted = prod.melt(id_vars=['main_id'], value_vars=metal_cols,
                       var_name='metal', value_name='mass_t')
    melted['metal'] = melted['metal'].str.replace('_t', '', regex=False)

    # keep only nonzero, non-null metals
    melted = melted[melted['mass_t'].notna() & (melted['mass_t'] > 0)].copy()

    # ---- MASS ALLOCATION ----
    if allocation == 'mass':
        melted['allocation_factor'] = melted.groupby('main_id')['mass_t'] \
                                            .transform(lambda x: x / x.sum())

    # ---- ECONOMIC ALLOCATION ----
    elif allocation == 'economic':
        if price_df is None:
            raise ValueError("price_df must be provided for economic allocation.")

        # normalize price_df column names
        price_df = price_df.rename(columns=lambda x: x.strip().lower())
        if not {'commodity', 'price'}.issubset(price_df.columns):
            raise ValueError("price_df must contain ['commodity', 'price'] columns.")

        melted = melted.merge(price_df[['commodity', 'price']],
                              left_on='metal', right_on='commodity', how='left')

        melted['mass_value'] = melted['mass_t'] * melted['price']
        melted['allocation_factor'] = melted.groupby('main_id')['mass_value'] \
                                            .transform(lambda x: x / x.sum())
    else:
        raise ValueError("allocation must be 'mass' or 'economic'.")

    # merge with flows
    out = df.merge(melted[['main_id', 'metal', 'mass_t', 'allocation_factor']],
                   on='main_id', how='inner')

    out['value_normalized'] = (out[value_col] / out['mass_t']) * out['allocation_factor']
    out['functional_unit'] = out['metal'] + ', usable ore'
    out['normalization_key'] = f"metal_{allocation}"

    return out


class LCIDatabaseBuilder:
    """
    A class to build and populate Brightway2 LCI databases from DataFrames.
    """

    def __init__(self, db_name, project_name='metallican'):
        """
        Initialize the builder with a database name and Brightway project.
        Automatically sets or creates the project and registers the database.
        """
        self.project_name = project_name
        self.db_name = db_name

        # Ensure we are working in the right Brightway project
        bw.projects.set_current(self.project_name)
        print(f"📂 Active Brightway project: {bw.projects.current}")

        # Initialize or register the database
        if self.db_name not in bw.databases:
            self.db = bw.Database(self.db_name)
            self.db.register()
            print(f"🆕 Database '{self.db_name}' created.")
        else:
            self.db = bw.Database(self.db_name)
            print(f"✅ Using existing database '{self.db_name}'.")

        # Container for the LCI entries
        self.lcis = {}


    def build_lci_entries(self, df): # Method of the LCIDatabaseBuilder
        """
        Initialize LCI activities from a DataFrame.
        Each activity automatically gets a production exchange (its reference product).
        """

        self.lcis = {}

        for _, row in df.iterrows():
            site_id = row['site_id']
            name = row['activity_name']
            product = row['functional_unit'] # reference product definition
            location = CA_provinces.get(row.get('province', ''), 'CA')

            # Brightway activity key
            unique_code = f'{site_id}_{product}'
            key = (self.db_name, unique_code) # Brightway key

            # Commodities given by NRCan (for use in the description
            nrcan_commodities = row.get('commodities')

            # Create process entry
            self.lcis[key] = {
                'name': name,
                'unit': 'kilogram',
                'location': location,
                'reference product': product,
                'exchanges': [
                    {
                        'input': key,  # self-reference
                        'amount': 1.0,
                        'unit': 'kilogram',
                        'type': 'production',
                        'name': product,
                        'product': product,
                        'reference product': product,
                    }
                ],
                'type': 'process',
                'comment': (
                    f"This is a site-specific LCI drawn from the MetalliCan database. Site ID is {site_id}. NRCan reports for this list of commodities: {nrcan_commodities}. Production data were only found for: {product}."
                )
            }

        print(f"✅ Created {len(self.lcis)} base LCI activities with production exchanges.")
        return self.lcis


    def populate_technosphere_exchanges(self, technosphere_df):
        """
        Populate technosphere exchanges per (site_id + functional_unit).
        Expected columns:
            ['site_id', 'functional_unit', 'Database', 'Activity',
             'Product', 'Amount', 'Unit', 'Location']
        """
        print("⚙️ Populating technosphere exchanges")

        # --- 1️⃣ Cache all databases ---
        db_lookup = {}
        for db_name in technosphere_df["Database"].dropna().unique():
            try:
                db = bw.Database(db_name)
                lookup = {(act["name"], act.get("location", None)): act.key for act in db}
                db_lookup[db_name] = lookup
                print(f"   ✅ Cached {len(lookup)} activities from {db_name}")
            except Exception as e:
                print(f"   ⚠️ Could not load database '{db_name}': {e}")

        # --- 2️⃣ Populate ---
        missing_keys, added = [], 0

        for key, process in self.lcis.items():
            # Retrieve metadata from comment
            comment = process["comment"]
            site_id = comment.split("Site ID is ")[-1].split(". NRCan")[0].strip()
            product = process.get("reference product")

            # Filter for this (site_id + functional_unit)
            site_exchanges = technosphere_df[
                (technosphere_df["site_id"].astype(str) == site_id)
                & (technosphere_df["functional_unit"].astype(str) == str(product))
                ]
            if site_exchanges.empty:
                continue

            for _, row in site_exchanges.iterrows():
                db_name = row["Database"]
                act_name = row["Activity"]
                loc = row.get("Location", None)
                lookup = db_lookup.get(db_name, {})
                input_key = lookup.get((act_name, loc)) or lookup.get((act_name, None))

                if not input_key:
                    missing_keys.append((db_name, act_name, loc))
                    continue

                exchange = {
                    "input": input_key,
                    "amount": float(row["Amount"]),
                    "unit": row["Unit"],
                    "type": "technosphere",
                    "name": act_name,
                    "product": row.get("Product", None),
                    "location": loc,
                }
                process["exchanges"].append(exchange)
                added += 1

        print(f"✅ Added {added} technosphere exchanges.")
        if missing_keys:
            print(f"⚠️ {len(missing_keys)} exchanges could not be matched:")
            for db, act, loc in missing_keys[:10]:
                print(f"   - {act} ({db}, {loc})")
            if len(missing_keys) > 10:
                print(f"   ... and {len(missing_keys) - 10} more.")

    def populate_biosphere_exchanges(self, biosphere_df):
        """
        Populate biosphere exchanges per (site_id + functional_unit).
        Expected columns:
            ['site_id', 'functional_unit', 'Database', 'Flow Name',
             'Compartments', 'Amount', 'Unit']
        """
        print("🌱 Populating biosphere exchanges")

        db_lookup = {}
        for db_name in biosphere_df["Database"].dropna().unique():
            try:
                db = bw.Database(db_name)
                lookup = {}
                for flow in db:
                    comp_tuple = tuple(flow.get("categories", []))
                    lookup[(flow["name"], comp_tuple)] = flow.key
                db_lookup[db_name] = lookup
                print(f"   ✅ Cached {len(lookup)} biosphere flows from {db_name}")
            except Exception as e:
                print(f"   ⚠️ Could not load biosphere database '{db_name}': {e}")

        missing_keys, added = [], 0

        for key, process in self.lcis.items():
            comment = process["comment"]
            site_id = comment.split("Site ID is ")[-1].split(". NRCan")[0].strip()
            product = process.get("reference product")

            site_exchanges = biosphere_df[
                (biosphere_df["site_id"].astype(str) == site_id)
                & (biosphere_df["functional_unit"].astype(str) == str(product))
                ]
            if site_exchanges.empty:
                continue

            for _, row in site_exchanges.iterrows():
                db_name = row["Database"]
                flow_name = row["Flow Name"]
                comps = row.get("Compartments", None)
                comps_tuple = tuple(str(c).strip() for c in comps.split("/")) if isinstance(comps, str) else ()
                lookup = db_lookup.get(db_name, {})

                input_key = lookup.get((flow_name, comps_tuple)) or next(
                    (v for (fname, _), v in lookup.items() if fname == flow_name),
                    None
                )
                if not input_key:
                    missing_keys.append((db_name, flow_name, comps))
                    continue

                exchange = {
                    "input": input_key,
                    "amount": float(row["Amount"]),
                    "unit": row["Unit"],
                    "type": "biosphere",
                    "name": flow_name,
                }
                process["exchanges"].append(exchange)
                added += 1

        print(f"✅ Added {added} biosphere exchanges.")
        if missing_keys:
            print(f"⚠️ {len(missing_keys)} biosphere flows could not be matched:")
            for db, name, comp in missing_keys[:10]:
                print(f"   - {name} ({comp}, {db})")
            if len(missing_keys) > 10:
                print(f"   ... and {len(missing_keys) - 10} more.")


    def consolidate_exchanges(self, by=("input", "unit", "type")):
        """
        Merge duplicate exchanges per activity by summing amounts.

        Parameters
        ----------
        by : tuple[str]
            Fields to group by when consolidating. Defaults to ('input','unit','type').
            You can add 'name' if you want to keep distinct labels separate.

        Notes
        -----
        - Assumes amounts are in the same unit. If not, normalize units before calling.
        - Keeps metadata from the first occurrence in each group.
        """
        total_before = 0
        total_after = 0
        for key, act_data in self.lcis.items():
            exchs = act_data.get("exchanges", [])
            total_before += len(exchs)
            buckets = {}
            first_meta = {}

            for exc in exchs:
                # build grouping key safely
                gk = tuple(exc.get(field) for field in by)
                amt = float(exc.get("amount", 0.0))

                buckets[gk] = buckets.get(gk, 0.0) + amt
                if gk not in first_meta:
                    first_meta[gk] = exc  # keep the first metadata exemplar

            # rebuild consolidated list
            new_exchs = []
            for gk, summed_amt in buckets.items():
                base = first_meta[gk].copy()
                base["amount"] = summed_amt
                new_exchs.append(base)

            act_data["exchanges"] = new_exchs
            total_after += len(new_exchs)

        print(f"🧮 Consolidation: {total_before} → {total_after} exchanges (summed duplicates).")


    def write_to_database(self, overwrite=True):
        """
        Write all activities and exchanges to the Brightway2 database.
        """
        print(f"🧱 Writing {len(self.lcis)} activities to database '{self.db_name}'...")

        for key, act_data in self.lcis.items():
            act_code = key[1]

            # Handle overwrite safely
            if overwrite and (self.db_name, act_code) in self.db:
                print(f"♻️ Overwriting existing activity: {act_code}")
                self.db.delete((self.db_name, act_code))

            act = self.db.new_activity(
                code=act_code,
                **{k: v for k, v in act_data.items() if k != 'exchanges'}
            )
            act.save()

            for exc in act_data['exchanges']:
                try:
                    act.new_exchange(**exc).save()
                except Exception as e:
                    print(f"⚠️ Failed to save exchange for {act_data['name']}: {e}")

        # Only process after all exchanges exist
        try:
            self.db.process()
            print(f"✅ Database '{self.db_name}' processed successfully with {len(self.db)} activities.")
        except Exception as e:
            print(f"⚠️ Processing failed: {e}")


    def verify_database(self):
        """
        Print a summary of the database contents.
        """
        for act in self.db:
            print(act.key, act.as_dict())
