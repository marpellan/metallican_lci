import brightway2 as bw
from core.constants import CA_provinces

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


    def build_lci_entries(self, df, facility_col, site_id_col): # Method of the LCIDatabaseBuilder
        """
        Initialize LCI activities from a DataFrame.
        Each activity automatically gets a production exchange (its reference product).
        """

        self.lcis = {}

        for _, row in df.iterrows():
            name = f"{row['mining_processing_type']} {row['commodities']}, {row[facility_col]}".strip()
            location = CA_provinces.get(row.get('province', ''), 'CA')
            site_id = row[site_id_col]

            key = (self.db_name, name)

            # Reference product definition
            product = row.get('commodities')

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
                ## Create a unique code??
                'comment': (
                    f"This is a site-specific LCI drawn from the MetalliCan database. "
                    f"Site ID is {site_id}."
                )
            }

        print(f"✅ Created {len(self.lcis)} base LCI activities with production exchanges.")
        return self.lcis


    def populate_technosphere_exchanges(self, technosphere_df, site_id_column='main_id'):
        """
        Populate technosphere exchanges for all activities based on a DataFrame
        structured like Brightway technosphere exports.
        Expected columns: ['main_id', 'Activity', 'Product', 'Database', 'Amount', 'Unit']
        """
        print("Populating biosphere exchanges")

        # --- 1. Build lookup dictionaries once per database
        db_lookup = {}
        for db_name in technosphere_df['Database'].unique():
            try:
                db = bw.Database(db_name)
                lookup = {}
                for act in db:
                    key = (act['name'], act.get('location', None))
                    lookup[key] = act.key
                db_lookup[db_name] = lookup
                print(f"   ✅ Cached {len(lookup)} activities from {db_name}")
            except Exception as e:
                print(f"   ⚠️ Could not load database '{db_name}': {e}")

        # --- 2. Populate exchanges
        missing_keys = []
        count = 0
        for key, process in self.lcis.items():
            site_id = process['comment'].split('Site ID is ')[-1].strip('. ')
            site_exchanges = technosphere_df[technosphere_df[site_id_column] == site_id]

            for _, row in site_exchanges.iterrows():
                db_name = row['Database']
                act_name = row['Activity']
                loc = row.get('Location', None)

                # Look up the pre-cached key
                lookup = db_lookup.get(db_name, {})
                input_key = lookup.get((act_name, loc)) or lookup.get((act_name, None))

                if not input_key:
                    missing_keys.append((db_name, act_name, loc))
                    continue

                exchange = {
                    'input': input_key,
                    'amount': float(row['Amount']),
                    'unit': row['Unit'],
                    'type': 'technosphere',
                    'name': act_name,
                    'product': row.get('Product', None),
                    'location': loc
                }
                process['exchanges'].append(exchange)
                count += 1

        print(f"✅ Added {count} technosphere exchanges.")
        if missing_keys:
            print(f"⚠️ {len(missing_keys)} exchanges could not be matched:")
            for db, act, loc in missing_keys[:10]:  # show first 10 only
                print(f"   - {act} ({db}, {loc})")
            if len(missing_keys) > 10:
                print(f"   ... and {len(missing_keys) - 10} more.")


    def populate_biosphere_exchanges(self, biosphere_df, site_id_column="main_id"):
        """
        Efficiently populate biosphere exchanges for all activities.
        Expected columns in biosphere_df:
        ['main_id', 'Flow Name', 'Compartments', 'Database', 'Amount', 'Unit']
        """

        print("🌱 Populating biosphere exchanges")

        # --- 1️⃣ Build lookup cache for biosphere databases
        db_lookup = {}
        for db_name in biosphere_df['Database'].unique():
            try:
                db = bw.Database(db_name)
                lookup = {}
                for act in db:
                    # store by (flow name, compartment tuple)
                    comp_tuple = tuple(act.get("categories", []))
                    lookup[(act["name"], comp_tuple)] = act.key
                db_lookup[db_name] = lookup
                print(f"   ✅ Cached {len(lookup)} biosphere flows from {db_name}")
            except Exception as e:
                print(f"   ⚠️ Could not load biosphere database '{db_name}': {e}")

        # --- 2️⃣ Iterate through each activity in your LCI
        missing_keys = []
        added = 0

        for key, process in self.lcis.items():
            site_id = process['comment'].split('Site ID is ')[-1].strip('. ')
            site_exchanges = biosphere_df[biosphere_df[site_id_column] == site_id]

            for _, row in site_exchanges.iterrows():
                db_name = row['Database']
                flow_name = row['Flow Name']
                comps = row.get('Compartments', None)
                comps_tuple = tuple(str(c).strip() for c in comps.split('/')) if isinstance(comps, str) else ()

                lookup = db_lookup.get(db_name, {})
                # Try exact match, then fallback ignoring compartments
                input_key = lookup.get((flow_name, comps_tuple)) or next(
                    (v for (fname, _), v in lookup.items() if fname == flow_name),
                    None
                )

                if not input_key:
                    missing_keys.append((db_name, flow_name, comps))
                    continue

                exchange = {
                    'input': input_key,
                    'amount': float(row['Amount']),
                    'unit': row['Unit'],
                    'type': 'biosphere',
                    'name': flow_name,
                }
                process['exchanges'].append(exchange)
                added += 1

        print(f"✅ Added {added} biosphere exchanges.")
        if missing_keys:
            print(f"⚠️ {len(missing_keys)} flows could not be matched to biosphere3:")
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
