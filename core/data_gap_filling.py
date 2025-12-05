import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import math


@dataclass
class InferenceEngine:
    """
    Central container for inference.

    Holds core dataframes and exposes high-level methods like
    `infer_energy_for_sites(...)` that delegate to specialized
    inference classes (Energy, Land, Materials, etc.).
    """
    site_df: Optional[pd.DataFrame]          # not necessary
    production_df: pd.DataFrame    # for production, e.g. ore processed
    energy_df: pd.DataFrame
    co2_df: pd.DataFrame
    land_df: pd.DataFrame
    material_df: pd.DataFrame

    # We can attach sub-inference objects here
    energy_inference: "EnergyInference" = field(init=False)
    land_inference: "LandInference" = field(init=False)

    def __post_init__(self):
        self.energy_inference = EnergyInference(self)
        self.land_inference = LandInference(self)

    def infer_energy_for_sites(
            self,
            site_ids,
            ef_co2_per_unit,
            stationary_share_rules,
            default_shares,
            overwrite=False
    ):
        inferred = self.energy_inference.infer_for_sites(
            site_ids=site_ids,
            ef_co2_per_unit=ef_co2_per_unit,
            stationary_share_rules=stationary_share_rules,
            default_shares=default_shares,
            overwrite=overwrite
        )

        # append to main energy_df
        combined = pd.concat([self.energy_df, inferred], ignore_index=True)
        self.energy_df = combined.copy()

        return combined, inferred

    def infer_land_for_sites(
            self,
            site_ids,
            formula_open_pit="0.791 * ore_processed_t - 7.76e5",
            formula_underground="Uniform(UG_land_min, UG_land_max)",
            formula_other="Uniform(other_land_min, other_land_max)",
            overwrite=False
    ):
        inferred = self.land_inference.infer_for_sites(
            site_ids=site_ids,
            formula_open_pit=formula_open_pit,
            formula_underground=formula_underground,
            formula_other=formula_other,
            overwrite=overwrite
        )

        combined = pd.concat([self.land_df, inferred], ignore_index=True)
        self.land_df = combined.copy()

        return combined, inferred



class EnergyInference:
    """
    Infer fuel consumption (in original units L or m3) from CO2 emissions.
    Supports:
    - On-site transportation -> Diesel only
    - Stationary fuel combustion -> split into Diesel / NG / LPG based on rules

    Output columns match existing energy_df:
    ['site_id','activity_name','mining_processing_type','archetypes',
     'flow_type','subflow_type','value','unit','data_source','value_formula']
    """

    def __init__(self, engine):
        self.engine = engine  # reference to InferenceEngine

    # ---------- Helper: CO2 → fuel units ----------
    @staticmethod
    def co2_to_fuel_units(co2_tonnes, ef_g_per_unit):
        """
        Convert CO2 (tonnes) to fuel amount in original units (L or m3).
        """
        if co2_tonnes is None or pd.isna(co2_tonnes):
            return float("nan"), None

        co2_g = co2_tonnes * 1_000_000
        fuel_units = co2_g / ef_g_per_unit

        formula = f"({co2_tonnes} * 1e6) / {ef_g_per_unit}"

        return fuel_units, formula

    # ---------- Helper: choose stationary shares ----------
    def resolve_stationary_shares(self, site_row, stationary_share_rules, default_shares):
        """
        Determine which fuel shares apply based on mining_processing_type or archetype.
        """
        mtype = site_row.get("mining_processing_type", None)
        arche = site_row.get("archetypes", None)

        # 1) match mining_processing_type if rules exist
        if mtype in stationary_share_rules:
            return stationary_share_rules[mtype]

        # 2) match archetype
        if arche in stationary_share_rules:
            return stationary_share_rules[arche]

        # 3) default
        return default_shares

    # ---------- Main inference ----------
    def infer_for_sites(
        self,
        site_ids,
        ef_co2_per_unit,       # dict: {"diesel": g/L, "natural_gas": g/m3, "lpg": g/L}
        stationary_share_rules, # dict of dicts: rules by mining_processing_type or archetype
        default_shares,         # fallback shares
        overwrite=False
    ):

        records = []

        required_cols = {
            "site_id", "activity_name", "mining_processing_type",
            "archetypes", "release_pathway", "value"
        }
        missing = required_cols - set(self.engine.co2_df.columns)
        if missing:
            raise ValueError(f"Missing columns in CO2 df: {missing}")

        # Existing flows already present in energy_df
        existing_pairs = set(
            zip(self.engine.energy_df["site_id"], self.engine.energy_df["subflow_type"])
        )

        for site in site_ids:
            df_site = self.engine.co2_df[self.engine.co2_df["site_id"] == site]
            if df_site.empty:
                continue

            # Metadata for this site
            meta = df_site.iloc[0]
            activity_name = meta["activity_name"]
            mtype = meta["mining_processing_type"]
            arche = meta["archetypes"]

            # Aggregate CO2 by pathway
            transport_co2 = df_site.loc[
                df_site["release_pathway"] == "On-site Transportation", "value"
            ].sum()

            stationary_co2 = df_site.loc[
                df_site["release_pathway"] == "Stationary Fuel Combustion", "value"
            ].sum()

            # -------------------------------------------
            # 1) On-site transportation -> Diesel only
            # -------------------------------------------
            if not pd.isna(transport_co2) and transport_co2 > 0:

                subflow = "Diesel|Transport"
                if overwrite or (site, subflow) not in existing_pairs:

                    # Convert CO2 → fuel (L diesel)
                    fuel_units, formula_fuel = self.co2_to_fuel_units(
                        transport_co2, ef_co2_per_unit["diesel"]
                    )

                    rec = {
                        "site_id": site,
                        "activity_name": activity_name,
                        "mining_processing_type": mtype,
                        "archetypes": arche,
                        "flow_type": "Energy",
                        "subflow_type": subflow,
                        "value": fuel_units,
                        "unit": "L",
                        "data_source": "Inference from CO2 (transport diesel)",
                        "value_formula": formula_fuel,
                    }
                    records.append(rec)

            # -------------------------------------------
            # 2) Stationary fuel combustion
            # -------------------------------------------
            if not pd.isna(stationary_co2) and stationary_co2 > 0:

                # Determine which shares apply to this site
                shares = self.resolve_stationary_shares(meta, stationary_share_rules, default_shares)
                diesel_share = shares["diesel"]
                ng_share = shares["natural_gas"]
                lpg_share = shares["lpg"]

                # ---- Diesel (stationary) ----
                subflow = "Diesel|Stationary"
                if overwrite or (site, subflow) not in existing_pairs:

                    co2_diesel = stationary_co2 * diesel_share
                    fuel_units, formula_fuel = self.co2_to_fuel_units(
                        co2_diesel, ef_co2_per_unit["diesel"]
                    )

                    rec = {
                        "site_id": site,
                        "activity_name": activity_name,
                        "mining_processing_type": mtype,
                        "archetypes": arche,
                        "flow_type": "Energy",
                        "subflow_type": subflow,
                        "value": fuel_units,
                        "unit": "L",
                        "data_source": "Inference from CO2 (stationary diesel)",
                        "value_formula": formula_fuel,
                    }
                    records.append(rec)

                # ---- Natural gas ----
                subflow = "Natural gas|Stationary"
                if overwrite or (site, subflow) not in existing_pairs:

                    co2_ng = stationary_co2 * ng_share
                    fuel_units, formula_fuel = self.co2_to_fuel_units(
                        co2_ng, ef_co2_per_unit["natural_gas"]
                    )

                    rec = {
                        "site_id": site,
                        "activity_name": activity_name,
                        "mining_processing_type": mtype,
                        "archetypes": arche,
                        "flow_type": "Energy",
                        "subflow_type": subflow,
                        "value": fuel_units,
                        "unit": "m3",
                        "data_source": "Inference from CO2 (stationary natural gas)",
                        "value_formula": formula_fuel,
                    }
                    records.append(rec)

                # ---- LPG ----
                subflow = "LPG|Stationary"
                if overwrite or (site, subflow) not in existing_pairs:

                    co2_lpg = stationary_co2 * lpg_share
                    fuel_units, formula_fuel = self.co2_to_fuel_units(
                        co2_lpg, ef_co2_per_unit["lpg"]
                    )

                    rec = {
                        "site_id": site,
                        "activity_name": activity_name,
                        "mining_processing_type": mtype,
                        "archetypes": arche,
                        "flow_type": "Energy",
                        "subflow_type": subflow,
                        "value": fuel_units,
                        "unit": "L",
                        "data_source": "Inference from CO2 (stationary LPG)",
                        "value_formula": formula_fuel,
                    }
                    records.append(rec)

        # Return empty df if none inferred
        if not records:
            return pd.DataFrame(columns=self.engine.energy_df.columns)

        inferred = pd.DataFrame(records)

        # Reindex to match energy_df EXACT columns
        missing_cols = [c for c in self.engine.energy_df.columns if c not in inferred.columns]
        for c in missing_cols:
            inferred[c] = pd.NA

        inferred = inferred[self.engine.energy_df.columns]

        return inferred


class LandInference:
    """
    Infer land transformation area (m2) or land occupation formula
    based on mining_processing_type and production information.

    Output columns MUST match land_df:
    ['site_id','activity_name','mining_processing_type','archetypes',
     'value','operation_periods','unit','data_source','value_formula']
    """

    def __init__(self, engine):
        self.engine = engine  # reference to InferenceEngine

    # -----------------------------
    # Helper: identify mining type
    # -----------------------------
    @staticmethod
    def _is_open_pit(mtype):
        if not isinstance(mtype, str):
            return False
        m = mtype.lower()
        return ("open-pit" in m) or ("open pit" in m)

    @staticmethod
    def _is_underground(mtype):
        if not isinstance(mtype, str):
            return False
        return "underground" in mtype.lower()

    # -----------------------------
    # Main inference function
    # -----------------------------
    def infer_for_sites(
        self,
        site_ids,
        formula_open_pit="0.791 * ore_processed_t - 7.76e5",
        formula_underground="Uniform(UG_land_min, UG_land_max)",
        formula_other="Uniform(other_land_min, other_land_max)",
        overwrite=False,
    ):

        records = []

        # Existing entries
        existing_pairs = set(zip(
            self.engine.land_df["site_id"],
            self.engine.land_df["activity_name"]
        ))

        for site in site_ids:

            # Metadata from site_df
            site_row = self.engine.site_df[self.engine.site_df["site_id"] == site]
            if site_row.empty:
                print(f"⚠️ No site metadata for {site}")
                continue

            site_meta = site_row.iloc[0]
            activity_name = site_meta.get("activity_name")
            mtype = site_meta.get("mining_processing_type")
            arche = site_meta.get("archetypes")

            # Avoid overwriting existing values
            if not overwrite and (site, activity_name) in existing_pairs:
                continue

            # Fetch ore processed
            prod_row = self.engine.production_df[self.engine.production_df["site_id"] == site]
            ore_processed = None
            if not prod_row.empty:
                ore_processed = prod_row.iloc[0].get("ore_processed_t")

            # -------------------------
            # CASE A: Open-pit mines
            # -------------------------
            if self._is_open_pit(mtype):

                if ore_processed is not None and not pd.isna(ore_processed):
                    numeric_value = 0.791 * ore_processed - 7.76e5
                    value_formula = f"(0.791 * {ore_processed}) - 7.76e5"
                else:
                    numeric_value = None
                    value_formula = formula_open_pit

                rec = {
                    "site_id": site,
                    "activity_name": activity_name,
                    "mining_processing_type": mtype,
                    "archetypes": arche,
                    "value": numeric_value,
                    "operation_periods": None,
                    "unit": "m2",
                    "data_source": "Inference (open-pit model)",
                    "value_formula": value_formula
                }

                records.append(rec)
                continue

            # -------------------------
            # CASE B: Underground mines
            # -------------------------
            if self._is_underground(mtype):

                rec = {
                    "site_id": site,
                    "activity_name": activity_name,
                    "mining_processing_type": mtype,
                    "archetypes": arche,
                    "value": None,  # symbolic
                    "operation_periods": None,
                    "unit": "m2",
                    "data_source": "Inference (underground range)",
                    "value_formula": formula_underground
                }

                records.append(rec)
                continue

            # -------------------------
            # CASE C: Everything else
            # -------------------------

            rec = {
                "site_id": site,
                "activity_name": activity_name,
                "mining_processing_type": mtype,
                "archetypes": arche,
                "value": None,
                "operation_periods": None,
                "unit": "m2",
                "data_source": "Inference (other facilities range)",
                "value_formula": formula_other
            }

            records.append(rec)

        # -------------------------
        # Final assembly
        # -------------------------
        if not records:
            return pd.DataFrame(columns=self.engine.land_df.columns)

        inferred_df = pd.DataFrame(records)

        # Ensure exact column order
        for col in self.engine.land_df.columns:
            if col not in inferred_df:
                inferred_df[col] = pd.NA

        inferred_df = inferred_df[self.engine.land_df.columns]

        return inferred_df


