import brightway2 as bw
import pandas as pd
import re
from utils.constants import CA_provinces


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

                    # --- Amount selection logic ---
                if pd.notna(row.get("Amount")):
                    amount = float(row["Amount"])
                elif pd.notna(row.get("Amount_mean")):
                    amount = float(row["Amount_mean"])
                else:
                    print(
                            f"⚠️ Missing Amount & Amount_mean "
                            f"(site_id={site_id}, activity={act_name})"
                        )
                    continue

                exchange = {
                    "input": input_key,
                    "amount": amount,
                    "unit": row["Unit"],
                    "type": "technosphere",
                    "name": act_name,
                    "product": row.get("Product", None),
                    "location": loc,
                    "comment": "mean used (gap-filled)" if pd.isna(row["Amount"]) else "reported"
                    # add comment but no parenthesis
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


    # def build_market_activities(
    #     self,
    #     market_df,
    #     location="CA",
    #     unit="kilogram",
    #     market_prefix="Market for ",
    #     code_prefix="market_",
    #     strict=True,
    #     epsilon=0.3,
    # ):
    #     """
    #     Create market activities (ecoinvent-like) from a market share DataFrame.
    #
    #     Expected columns in market_df:
    #         - reference_product
    #         - site_id
    #         - market_share
    #     Optional columns:
    #         - activity_name (only used for logs/checks)
    #
    #     Assumes your site-specific activities codes are: f"{site_id}_{reference_product}"
    #     and are stored in self.lcis with key: (self.db_name, code)
    #     """
    #
    #     required = {"reference_product", "site_id", "market_share"}
    #     missing = required - set(market_df.columns)
    #     if missing:
    #         raise ValueError(f"Missing required columns in market_df: {missing}")
    #
    #     df = market_df.copy()
    #     df["reference_product"] = df["reference_product"].astype(str).str.strip()
    #     df["site_id"] = df["site_id"].astype(str).str.strip()
    #     df["market_share"] = pd.to_numeric(df["market_share"], errors="coerce")
    #
    #     if df["market_share"].isna().any():
    #         bad = df[df["market_share"].isna()]
    #         raise ValueError(f"Some market_share values are not numeric:\n{bad}")
    #
    #     def safe_code(s: str) -> str:
    #         """Make a simple Brightway-safe code."""
    #         s = str(s).strip()
    #         s = s.replace(" ", "_")
    #         # keep letters/numbers/_/-
    #         s = re.sub(r"[^A-Za-z0-9_\-]+", "", s)
    #         return s
    #
    #     created = 0
    #
    #     for rp, grp in df.groupby("reference_product"):
    #         share_sum = float(grp["market_share"].sum())
    #         if abs(share_sum - 1.0) > epsilon:
    #             print(f"⚠️ [{rp}] shares sum={share_sum:.6f}, renormalized to 1.0")
    #             grp = grp.copy()
    #             grp["market_share"] = grp["market_share"] / share_sum
    #             #if strict:
    #             #    raise ValueError(msg)
    #             #else:
    #             #    print("⚠️", msg)
    #
    #         # Market identifiers
    #         market_name = f"{market_prefix}{rp}"
    #         market_code = safe_code(f"{code_prefix}{rp}")
    #         market_key = (self.db_name, market_code)
    #
    #         # Production exchange (self reference)
    #         exchanges = [{
    #             "input": market_key,
    #             "amount": 1.0,
    #             "unit": unit,
    #             "type": "production",
    #             "name": rp,
    #             "product": rp,
    #             "reference product": rp,
    #         }]
    #
    #         # Technosphere inputs = market shares
    #         missing_suppliers = []
    #
    #         for _, row in grp.iterrows():
    #             site_id = row["site_id"]
    #             share = float(row["market_share"])
    #
    #             supplier_code = f"{site_id}_{rp}"
    #             supplier_key = (self.db_name, supplier_code)
    #
    #             if supplier_key not in self.lcis:
    #                 missing_suppliers.append(site_id)
    #                 continue
    #
    #             exchanges.append({
    #                 "input": supplier_key,
    #                 "amount": share,
    #                 "unit": unit,
    #                 "type": "technosphere",
    #                 "name": self.lcis[supplier_key]["name"],
    #                 "product": rp,
    #                 "location": self.lcis[supplier_key].get("location", None),
    #                 "comment": "market share",
    #             })
    #
    #         if missing_suppliers:
    #             msg = f"[{rp}] Missing site-specific activities for site_id: {missing_suppliers}"
    #             if strict:
    #                 raise KeyError(msg)
    #             else:
    #                 print("⚠️", msg)
    #
    #         # Add market activity to self.lcis
    #         self.lcis[market_key] = {
    #             "name": market_name,
    #             "unit": unit,
    #             "location": location,
    #             "reference product": rp,
    #             "exchanges": exchanges,
    #             "type": "market activity",
    #             "comment": f"Market activity built from site-specific LCIs (shares from market_df).",
    #         }
    #
    #         created += 1
    #
    #     print(f"🧩 Created {created} market activities.")
    #
    #     return created

    def build_market_activities(
            self,
            market_df,
            location="CA",
            unit="kilogram",
            market_prefix="Market for ",
            code_prefix="market_",
            strict=True,
            epsilon=0.3,
    ):
        required = {"reference_product", "site_id", "market_share"}
        missing = required - set(market_df.columns)
        if missing:
            raise ValueError(f"Missing required columns in market_df: {missing}")

        df = market_df.copy()
        df["reference_product"] = df["reference_product"].astype(str).str.strip()
        df["site_id"] = df["site_id"].astype(str).str.strip()
        df["market_share"] = pd.to_numeric(df["market_share"], errors="coerce")

        if df["market_share"].isna().any():
            bad = df[df["market_share"].isna()]
            raise ValueError(f"Some market_share values are not numeric:\n{bad}")

        def safe_code(s: str) -> str:
            s = str(s).strip().replace(" ", "_")
            s = re.sub(r"[^A-Za-z0-9_\-]+", "", s)
            return s

        created = 0

        for rp, grp in df.groupby("reference_product"):
            share_sum = float(grp["market_share"].sum())
            renorm = False
            if abs(share_sum - 1.0) > epsilon:
                print(f"⚠️ [{rp}] shares sum={share_sum:.6f}, renormalized to 1.0")
                grp = grp.copy()
                grp["market_share"] = grp["market_share"] / share_sum
                renorm = True

            market_name = f"{market_prefix}{rp}"
            market_code = safe_code(f"{code_prefix}{rp}")
            market_key = (self.db_name, market_code)

            # Minimal production exchange
            exchanges = [{
                "input": market_key,
                "amount": 1.0,
                "type": "production",
            }]

            missing_suppliers = []
            for _, row in grp.iterrows():
                site_id = row["site_id"]
                share = float(row["market_share"])

                supplier_code = f"{site_id}_{rp}"
                supplier_key = (self.db_name, supplier_code)

                if supplier_key not in self.lcis:
                    missing_suppliers.append(site_id)
                    continue

                # Minimal technosphere exchange
                exchanges.append({
                    "input": supplier_key,
                    "amount": share,
                    "type": "technosphere",
                })

            if missing_suppliers:
                msg = f"[{rp}] Missing site-specific activities for site_id: {missing_suppliers}"
                if strict:
                    raise KeyError(msg)
                else:
                    print("⚠️", msg)

            self.lcis[market_key] = {
                "name": market_name,
                "unit": unit,
                "location": location,
                "reference product": rp,
                "exchanges": exchanges,
                "type": "market activity",
                "comment": (
                    f"Market activity built from site-specific LCIs. "
                    f"n_sites={len(grp)}; renormalized={renorm}; original_sum={share_sum:.6f}."
                ),
            }

            created += 1

        print(f"🧩 Created {created} market activities.")
        return created

    # def write_to_database(self, overwrite=True):
    #     """
    #     Write all activities and exchanges to the Brightway2 database.
    #     """
    #     print(f"🧱 Writing {len(self.lcis)} activities to database '{self.db_name}'...")
    #
    #     for key, act_data in self.lcis.items():
    #         act_code = key[1]
    #
    #         # Handle overwrite safely
    #         if overwrite and (self.db_name, act_code) in self.db:
    #             print(f"♻️ Overwriting existing activity: {act_code}")
    #             self.db.delete((self.db_name, act_code))
    #
    #         act = self.db.new_activity(
    #             code=act_code,
    #             **{k: v for k, v in act_data.items() if k != 'exchanges'}
    #         )
    #         act.save()
    #
    #         for exc in act_data['exchanges']:
    #             try:
    #                 act.new_exchange(**exc).save()
    #             except Exception as e:
    #                 print(f"⚠️ Failed to save exchange for {act_data['name']}: {e}")
    #
    #     # Only process after all exchanges exist
    #     try:
    #         self.db.process()
    #         print(f"✅ Database '{self.db_name}' processed successfully with {len(self.db)} activities.")
    #     except Exception as e:
    #         print(f"⚠️ Processing failed: {e}")

    def write_to_database(self, overwrite=True):
        print(f"🧱 Writing {len(self.lcis)} activities to database '{self.db_name}'...")

        # Split site-specific vs markets based on 'type'
        site_keys = [k for k, v in self.lcis.items() if
                     v.get("type") in ("process", "process activity", "processes", "process")]
        market_keys = [k for k, v in self.lcis.items() if v.get("type") == "market activity"]
        other_keys = [k for k in self.lcis if k not in set(site_keys) | set(market_keys)]

        # Helper: create activity only (no exchanges)
        def _create_act(key):
            act_code = key[1]
            act_data = self.lcis[key]

            if overwrite and (self.db_name, act_code) in self.db:
                self.db.delete((self.db_name, act_code))

            act = self.db.new_activity(
                code=act_code,
                **{k: v for k, v in act_data.items() if k != "exchanges"}
            )
            act.save()

        # Helper: add exchanges for one activity
        def _add_exchanges(key, act_obj):
            for exc in self.lcis[key].get("exchanges", []):
                act_obj.new_exchange(**exc).save()

        # -------------------------
        # PASS A: create site acts
        # -------------------------
        for k in site_keys:
            _create_act(k)
        print(f"✅ Created {len(site_keys)} site-specific activities.")

        # -------------------------
        # PASS B: create markets
        # -------------------------
        for k in market_keys:
            _create_act(k)
        print(f"✅ Created {len(market_keys)} market activities.")

        # Create any remaining (optional)
        for k in other_keys:
            _create_act(k)

        # Build lookup of actual BW activity objects
        act_obj = {act.key: act for act in self.db}

        # -------------------------
        # PASS C: add exchanges
        # -------------------------
        exch_ok, exch_fail = 0, 0

        for k in site_keys + market_keys + other_keys:
            act = act_obj.get(k)
            if act is None:
                print(f"⚠️ Missing activity in DB for key={k}")
                continue
            try:
                _add_exchanges(k, act)
                exch_ok += len(self.lcis[k].get("exchanges", []))
            except Exception as e:
                exch_fail += 1
                print(f"⚠️ Failed exchanges for {self.lcis[k].get('name')} ({k}): {e}")

        print(f"✅ Saved exchanges: approx {exch_ok} (failures: {exch_fail})")

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


def export_bw_database_to_excel(db_name, out_xlsx):
    db = bd.Database(db_name)

    # ---------- Activities ----------
    acts_rows = []
    for act in db:
        d = act.as_dict()
        acts_rows.append({
            "database": act.key[0],
            "code": act.key[1],
            "name": d.get("name"),
            "reference_product": d.get("reference product") or d.get("product"),
            "location": d.get("location"),
            "unit": d.get("unit"),
            "type": d.get("type"),
            "comment": d.get("comment"),
        })
    activities_df = pd.DataFrame(acts_rows)

    # ---------- Exchanges ----------
    exc_rows = []
    for act in db:
        act_d = act.as_dict()
        act_name = act_d.get("name")
        act_rp = act_d.get("reference product") or act_d.get("product")

        for exc in act.exchanges():
            e = exc.as_dict()
            inp = e.get("input")
            out = e.get("output")

            # input/output sont des tuples (db, code)
            in_db, in_code = (inp if isinstance(inp, (tuple, list)) else (None, None))
            out_db, out_code = (out if isinstance(out, (tuple, list)) else (None, None))

            exc_rows.append({
                "output_db": out_db,
                "output_code": out_code,
                "output_name": act_name,
                "output_reference_product": act_rp,
                "type": e.get("type"),
                "amount": e.get("amount"),
                "unit": e.get("unit"),
                "input_db": in_db,
                "input_code": in_code,
                # champs optionnels
                "name": e.get("name"),
                "product": e.get("product"),
                "location": e.get("location"),
                "comment": e.get("comment"),
            })

    exchanges_df = pd.DataFrame(exc_rows)

    # ---------- Write ----------
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        activities_df.to_excel(writer, sheet_name="activities", index=False)
        exchanges_df.to_excel(writer, sheet_name="exchanges", index=False)

    #print("Wrote:", out_xlsx)
    return activities_df, exchanges_df
