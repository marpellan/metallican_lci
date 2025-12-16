import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import math
import re
import numpy as np


INFERRED_COLS = [
    "site_id",
    "activity_name",
    "mining_processing_type",
    "archetypes",
    "flow_type",
    "subflow_type",
    "value",                   # <- this will store the inferred mean
    "unit",
    "data_source",
    "value_formula",
    "parameter_distribution",
]

def mean_of_distribution(dist: str):
    """
    Returns the mean of a distribution string.
    Supported:
      - "Uniform(a, b)"
      - "Uniform(a, b) * Uniform(c, d)" (product of independent uniforms)
    Returns None if parsing fails.
    """
    if not isinstance(dist, str) or not dist.strip():
        return None

    matches = re.findall(r"Uniform\(([-+0-9.eE]+),\s*([-+0-9.eE]+)\)", dist)
    if not matches:
        return None

    means = []
    for a, b in matches:
        a = float(a); b = float(b)
        means.append((a + b) / 2)

    return float(np.prod(means))


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
    energy_inference: "EnergyInference" = field(init=False) # This InferenceEngine will have an EnergyInference object
    land_inference: "LandInference" = field(init=False)
    material_inference: "MaterialInference" = field(init=False)
    explosives_inference: "ExplosivesInference" = field(init=False)
    cement_inference: "CementInference" = field(init=False)

    def __post_init__(self):
        self.energy_inference = EnergyInference(self)
        self.land_inference = LandInference(self)
        #self.material_inference = MaterialInference(self) Needs external rules table, engine alone is not enough
        self.explosives_inference = ExplosivesInference(self)
        self .cement_inference = CementInference(self)

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

    def init_material_inference(self, material_rules_df):
        self.material_inference = MaterialInference(self, material_rules_df)

    def infer_material_for_sites(
            self,
            site_ids,
            overwrite=False,
            verbose=True,
    ):
        inferred = self.material_inference.infer_for_sites(
            site_ids=site_ids,
            overwrite=overwrite,
            verbose=verbose,
        )

        combined = pd.concat([self.material_df, inferred], ignore_index=True)
        self.material_df = combined.copy()

        return combined, inferred

    def infer_explosives_for_sites(
            self,
            site_ids,
            explosive_params,
            overwrite=False,
            verbose=True,
    ):
        inference = ExplosivesInference(self)

        inferred = inference.infer_for_sites(
            site_ids=site_ids,
            explosive_params=explosive_params,
            overwrite=overwrite,
            verbose=verbose,
        )

        combined = pd.concat([self.material_df, inferred], ignore_index=True)
        self.material_df = combined.copy()

        return combined, inferred

    def infer_cement_for_sites(
            self,
            site_ids,
            cement_params,
            overwrite=False,
            verbose=True,
    ):
        inferred = self.cement_inference.infer_for_sites(
            site_ids=site_ids,
            cement_params=cement_params,
            overwrite=overwrite,
            verbose=verbose,
        )

        self.material_df = pd.concat(
            [self.material_df, inferred],
            ignore_index=True
        )

        return self.material_df, inferred

    def mean_of_distribution(dist: str):
        """
        Return the mean of a distribution expressed as a string.
        Supports:
          - Uniform(a, b)
          - Products of Uniform(...) * Uniform(...)
        """
        if not isinstance(dist, str):
            return None

        matches = re.findall(r"Uniform\(([-+0-9.eE]+),\s*([-+0-9.eE]+)\)", dist)
        if not matches:
            return None

        means = [(float(a) + float(b)) / 2 for a, b in matches]
        return np.prod(means)


class EnergyInference:
    """
    Infer fuel consumption (in original units L or m3) from CO2 emissions.
    Supports:
    - On-site transportation -> Diesel only
    - Stationary fuel combustion -> split into Diesel / NG / LPG based on rules

    Output columns match enriched energy_df:
    ['site_id','activity_name','mining_processing_type','archetypes',
     'flow_type','subflow_type','value','unit','data_source','value_formula',
     'reference_flow','amount_parameter','parameter_distribution',
     'value_min','value_mean','value_max']
    """

    def __init__(self, engine):
        self.engine = engine

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
        mtype = site_row.get("mining_processing_type", None)
        arche = site_row.get("archetypes", None)

        if mtype in stationary_share_rules:
            return stationary_share_rules[mtype]
        if arche in stationary_share_rules:
            return stationary_share_rules[arche]
        return default_shares

    # ---------- Main inference ----------
    def infer_for_sites(
        self,
        site_ids,
        ef_co2_per_unit,        # e.g., {"diesel": 2681, "natural_gas": 2354, "lpg": 2753}
        stationary_share_rules, # e.g., rules by mtype or archetype
        default_shares,         # fallback shares
        overwrite=False,
        verbose=True,
    ):
        records = []

        required_cols = {
            "site_id", "activity_name", "mining_processing_type",
            "archetypes", "release_pathway", "value"
        }
        missing = required_cols - set(self.engine.co2_df.columns)
        if missing:
            raise ValueError(f"Missing columns in CO2 df: {missing}")

        existing_pairs = set(
            zip(self.engine.energy_df["site_id"], self.engine.energy_df["subflow_type"])
        )

        for site in site_ids:
            df_site = self.engine.co2_df[self.engine.co2_df["site_id"] == site]
            if df_site.empty:
                continue

            meta = df_site.iloc[0]
            activity_name = meta["activity_name"]
            mtype = meta["mining_processing_type"]
            arche = meta["archetypes"]

            transport_co2 = df_site.loc[
                df_site["release_pathway"] == "On-site Transportation", "value"
            ].sum()

            stationary_co2 = df_site.loc[
                df_site["release_pathway"] == "Stationary Fuel Combustion", "value"
            ].sum()

            # --- Transport Diesel ---
            if not pd.isna(transport_co2) and transport_co2 > 0:
                subflow = "Diesel|Transport"
                if overwrite or (site, subflow) not in existing_pairs:
                    val, formula = self.co2_to_fuel_units(transport_co2, ef_co2_per_unit["diesel"])
                    rec = {
                        "site_id": site,
                        "activity_name": activity_name,
                        "mining_processing_type": mtype,
                        "archetypes": arche,
                        "flow_type": "Energy",
                        "subflow_type": subflow,
                        "value": val,
                        "unit": "L",
                        "data_source": "Inference from CO2 (transport diesel)",
                        "value_formula": formula,
                        "reference_flow": "co2_emissions_t",
                        "amount_parameter": "co2_emissions_t",
                        "parameter_distribution": None,
                        "value_min": val,
                        "value_mean": val,
                        "value_max": val,
                    }
                    records.append(rec)

            # --- Stationary fuels ---
            if not pd.isna(stationary_co2) and stationary_co2 > 0:
                shares = self.resolve_stationary_shares(meta, stationary_share_rules, default_shares)

                for fuel in ["diesel", "natural_gas", "lpg"]:
                    subflow = f"{fuel.replace('_', ' ').title()}|Stationary"
                    if overwrite or (site, subflow) not in existing_pairs:
                        co2_share = stationary_co2 * shares[fuel]
                        val, formula = self.co2_to_fuel_units(co2_share, ef_co2_per_unit[fuel])
                        unit = "m3" if fuel == "natural_gas" else "L"
                        rec = {
                            "site_id": site,
                            "activity_name": activity_name,
                            "mining_processing_type": mtype,
                            "archetypes": arche,
                            "flow_type": "Energy",
                            "subflow_type": subflow,
                            "value": val,
                            "unit": unit,
                            "data_source": f"Inference from CO2 (stationary {fuel})",
                            "value_formula": formula,
                            "reference_flow": "co2_emissions_t",
                            "amount_parameter": "co2_emissions_t",
                            "parameter_distribution": None,
                            "value_min": val,
                            "value_mean": val,
                            "value_max": val,
                        }
                        records.append(rec)

        if not records:
            return pd.DataFrame(columns=self.engine.energy_df.columns)

        df = pd.DataFrame(records)

        # Ensure all expected columns are present
        for col in self.engine.energy_df.columns:
            if col not in df:
                df[col] = pd.NA

        return df[self.engine.energy_df.columns]


# class LandInference:
#     """
#     Infer land transformation area (m2) or land occupation formula
#     based on mining_processing_type and ore_processed_t.
#
#     Now returns symbolic formulas + min/mean/max estimates for Brightway use.
#     """
#
#     def __init__(self, engine):
#         self.engine = engine
#
#     @staticmethod
#     def _is_open_pit(mtype):
#         if not isinstance(mtype, str):
#             return False
#         m = mtype.lower()
#         return "open-pit" in m or "open pit" in m
#
#     @staticmethod
#     def _is_underground(mtype):
#         if not isinstance(mtype, str):
#             return False
#         return "underground" in mtype.lower()
#
#     @staticmethod
#     def _parse_uniform_range(formula_str):
#         """Extract min, mean, and max from Uniform(a, b) string."""
#         if not isinstance(formula_str, str):
#             return None, None, None
#         match = re.search(r"Uniform\(([^,]+),\s*([^)]+)\)", formula_str)
#         if match:
#             try:
#                 a = float(match.group(1))
#                 b = float(match.group(2))
#                 return a, (a + b) / 2, b
#             except ValueError:
#                 return None, None, None
#         return None, None, None
#
#     def infer_for_sites(
#         self,
#         site_ids,
#         formula_open_pit="0.791 * ore_processed_t - 7.76e5",
#         formula_underground="Uniform(800000, 1200000)",  # updated to numeric default
#         formula_other="Uniform(200000, 400000)",
#         overwrite=False,
#     ):
#         records = []
#
#         existing_pairs = set(zip(
#             self.engine.land_df["site_id"],
#             self.engine.land_df["activity_name"]
#         ))
#
#         for site_id in site_ids:
#             site_row = self.engine.site_df[self.engine.site_df["site_id"] == site_id]
#             if site_row.empty:
#                 print(f"⚠️ No site metadata for {site_id}")
#                 continue
#
#             site_meta = site_row.iloc[0]
#             activity_name = site_meta.get("activity_name")
#             mtype = site_meta.get("mining_processing_type")
#             archetype = site_meta.get("archetypes")
#
#             if not overwrite and (site_id, activity_name) in existing_pairs:
#                 continue
#
#             ore_row = self.engine.production_df[self.engine.production_df["site_id"] == site_id]
#             ore = ore_row.iloc[0]["ore_processed_t"] if not ore_row.empty else None
#
#             if self._is_open_pit(mtype):
#                 value_formula = formula_open_pit
#                 value_min = value_max = value_mean = None
#                 if ore is not None and pd.notna(ore):
#                     value_mean = 0.791 * ore - 7.76e5
#                     value_min = value_max = value_mean
#                     value_formula = f"(0.510 * {ore}) - 7.14e6"
#
#                 rec = {
#                     "site_id": site_id,
#                     "activity_name": activity_name,
#                     "mining_processing_type": mtype,
#                     "archetypes": archetype,
#                     "value": None,
#                     "unit": "m2",
#                     "operation_periods": None,
#                     "data_source": "Inference (open-pit model)",
#                     "value_formula": value_formula,
#                     "amount_parameter": "ore_processed_t",
#                     #"reference_flow": "ore_processed_t",
#                     "parameter_distribution": None,
#                     "value_min": value_min,
#                     "value_mean": value_mean,
#                     "value_max": value_max,
#                 }
#                 records.append(rec)
#                 continue
#
#             if self._is_underground(mtype):
#                 val_min, val_mean, val_max = self._parse_uniform_range(formula_underground)
#                 rec = {
#                     "site_id": site_id,
#                     "activity_name": activity_name,
#                     "mining_processing_type": mtype,
#                     "archetypes": archetype,
#                     "value": None,
#                     "unit": "m2",
#                     "operation_periods": None,
#                     "data_source": "Inference (underground range)",
#                     "value_formula": formula_underground,
#                     "amount_parameter": None,
#                     #"reference_flow": None,
#                     "parameter_distribution": formula_underground,
#                     "value_min": val_min,
#                     "value_mean": val_mean,
#                     "value_max": val_max,
#                 }
#                 records.append(rec)
#                 continue
#
#             # For other facilities
#             val_min, val_mean, val_max = self._parse_uniform_range(formula_other)
#             rec = {
#                 "site_id": site_id,
#                 "activity_name": activity_name,
#                 "mining_processing_type": mtype,
#                 "archetypes": archetype,
#                 "value": None,
#                 "unit": "m2",
#                 "operation_periods": None,
#                 "data_source": "Inference (other facilities range)",
#                 "value_formula": formula_other,
#                 "amount_parameter": None,
#                 #"reference_flow": None,
#                 "parameter_distribution": formula_other,
#                 "value_min": val_min,
#                 "value_mean": val_mean,
#                 "value_max": val_max,
#             }
#             records.append(rec)
#
#         columns = [
#             "site_id", "activity_name", "mining_processing_type", "archetypes",
#             "value", "unit", "operation_periods", "data_source", "value_formula",
#             "amount_parameter", "parameter_distribution",
#             "value_min", "value_mean", "value_max"
#         ]
#         return pd.DataFrame(records, columns=columns)
class LandInference:

    def __init__(self, engine):
        self.engine = engine

    @staticmethod
    def _is_open_pit(mtype):
        return isinstance(mtype, str) and ("open-pit" in mtype.lower() or "open pit" in mtype.lower())

    @staticmethod
    def _is_underground(mtype):
        return isinstance(mtype, str) and ("underground" in mtype.lower())

    def infer_for_sites(
        self,
        site_ids,
        formula_open_pit="0.791 * ore_processed_t - 7.76e5",
        formula_underground="Uniform(800000, 1200000)",
        formula_other="Uniform(200000, 400000)",
        overwrite=False,
    ):
        records = []
        existing_pairs = set(zip(
            self.engine.land_df.get("site_id", pd.Series(dtype=object)),
            self.engine.land_df.get("subflow_type", pd.Series(dtype=object)),
        ))

        for site_id in site_ids:
            site_row = self.engine.site_df[self.engine.site_df["site_id"] == site_id]
            if site_row.empty:
                continue

            meta = site_row.iloc[0]
            mtype = meta.get("mining_processing_type")

            prod_row = self.engine.production_df[self.engine.production_df["site_id"] == site_id]
            ore = prod_row.iloc[0].get("ore_processed_t") if not prod_row.empty else None

            subflow = "Land area"
            if not overwrite and (site_id, subflow) in existing_pairs:
                continue

            value_inferred = None
            dist = None

            if self._is_open_pit(mtype):
                # deterministic regression
                formula = formula_open_pit
                source = "Inference | land | open-pit regression"
                if ore is not None and pd.notna(ore):
                    # safe eval context
                    value_inferred = float(eval(formula, {"__builtins__": {}}, {"ore_processed_t": ore}))

            elif self._is_underground(mtype):
                dist = formula_underground
                formula = f"ore_processed_t * {dist}"
                source = "Inference | land | underground"
                mean_factor = mean_of_distribution(dist)
                if mean_factor is not None and ore is not None and pd.notna(ore):
                    value_inferred = ore * mean_factor

            else:
                dist = formula_other
                formula = f"ore_processed_t * {dist}"
                source = "Inference | land | other"
                mean_factor = mean_of_distribution(dist)
                if mean_factor is not None and ore is not None and pd.notna(ore):
                    value_inferred = ore * mean_factor

            records.append({
                "site_id": site_id,
                "activity_name": meta.get("activity_name"),
                "mining_processing_type": mtype,
                "archetypes": meta.get("archetypes"),
                "flow_type": "Land",
                "subflow_type": subflow,
                "value": value_inferred,  # ✅ inferred mean / deterministic estimate
                "unit": "m2",
                "data_source": source,
                "value_formula": formula,
                "amount_parameter": "ore_processed_t",
                "parameter_distribution": dist,
            })

        return pd.DataFrame(records, columns=INFERRED_COLS)


# class MaterialInference:
#     """
#     Infer material input flows based on archetype-driven rules, enriched with value range estimation.
#
#     Output includes:
#     - value_formula (symbolic)
#     - value_min, value_mean, value_max (numeric, resolved if ore_processed_t is known)
#     - parameter_distribution (optional)
#     """
#
#     def __init__(self, engine, material_rules_df):
#         self.engine = engine
#         self.material_rules_df = material_rules_df.copy()
#
#         required_cols = {
#             "archetype", "material_name", "flow_type", "value_formula", "unit"
#         }
#         missing = required_cols - set(self.material_rules_df.columns)
#         if missing:
#             raise ValueError(f"Material rules table missing columns: {missing}")
#
#     def _extract_distribution_stats(self, formula: str):
#         if not isinstance(formula, str):
#             return None, None, None, None
#
#         # Detect Uniform(...)
#         uniform_matches = re.findall(r"Uniform\(([-+eE0-9.]+),\s*([-+eE0-9.]+)\)", formula)
#         if uniform_matches:
#             param_dist = " * ".join([f"Uniform({a}, {b})" for a, b in uniform_matches])
#             param_min = np.prod([float(a) for a, _ in uniform_matches])
#             param_max = np.prod([float(b) for _, b in uniform_matches])
#             param_mean = np.prod([(float(a) + float(b)) / 2 for a, b in uniform_matches])
#             return param_dist, param_min, param_mean, param_max
#
#         # Detect 'a-b * ore_processed_t' style
#         range_match = re.match(r"\s*([0-9.eE+-]+)-([0-9.eE+-]+)\s*\*\s*", formula)
#         if range_match:
#             a, b = float(range_match[1]), float(range_match[2])
#             param_dist = f"Uniform({a}, {b})"
#             return param_dist, a, (a + b) / 2, b
#
#         return None, None, None, None
#
#     def infer_for_sites(self, site_ids, overwrite=False, verbose=True):
#         records = []
#
#         existing_pairs = set(zip(
#             self.engine.material_df["site_id"],
#             self.engine.material_df["subflow_type"]
#         ))
#
#         for site_id in site_ids:
#             site_row = self.engine.site_df[self.engine.site_df["site_id"] == site_id]
#             if site_row.empty:
#                 if verbose:
#                     print(f"⚠️ No site metadata for {site_id}")
#                 continue
#
#             site_meta = site_row.iloc[0]
#             archetype = site_meta.get("archetypes")
#             activity_name = site_meta.get("activity_name")
#             mining_type = site_meta.get("mining_processing_type")
#
#             if not archetype or pd.isna(archetype):
#                 if verbose:
#                     print(f"⚠️ No archetype for site {site_id}")
#                 continue
#
#             rules = self.material_rules_df[self.material_rules_df["archetype"] == archetype]
#             if rules.empty:
#                 if verbose:
#                     print(f"⚠️ No material rules for archetype '{archetype}'")
#                 continue
#
#             # Get ore_processed_t from production_df
#             prod_row = self.engine.production_df[self.engine.production_df["site_id"] == site_id]
#             ore_processed = None
#             if not prod_row.empty:
#                 ore_processed = prod_row.iloc[0].get("ore_processed_t")
#
#             for _, rule in rules.iterrows():
#                 subflow = rule["material_name"]
#                 if not overwrite and (site_id, subflow) in existing_pairs:
#                     continue
#
#                 formula = rule["value_formula"]
#                 param_dist, param_min, param_mean, param_max = self._extract_distribution_stats(formula)
#
#                 # Multiply by ore if possible
#                 value_min = value_mean = value_max = None
#                 if ore_processed and param_min is not None:
#                     value_min = ore_processed * param_min
#                     value_mean = ore_processed * param_mean
#                     value_max = ore_processed * param_max
#
#                 rec = {
#                     "site_id": site_id,
#                     "activity_name": activity_name,
#                     "mining_processing_type": mining_type,
#                     "archetypes": archetype,
#                     "flow_type": rule.get("flow_type", "Material"),
#                     "subflow_type": subflow,
#                     "value": None,
#                     "unit": rule["unit"],
#                     "data_source": (
#                         f"Archetype inference ({rule.get('process', '')}) | "
#                         f"{rule.get('source', '')}"
#                     ).strip(),
#                     "value_formula": formula,
#                     "amount_parameter": None,
#                     "parameter_distribution": param_dist,
#                     "value_min": value_min,
#                     "value_mean": value_mean,
#                     "value_max": value_max,
#                     #"reference_flow": "ore_processed_t"
#                 }
#
#                 records.append(rec)
#
#         if not records:
#             return pd.DataFrame(columns=[
#                 "site_id", "activity_name", "mining_processing_type", "archetypes",
#                 "flow_type", "subflow_type", "value", "unit", "data_source",
#                 "value_formula", "amount_parameter", "parameter_distribution",
#                 "value_min", "value_mean", "value_max"
#             ])
#
#         return pd.DataFrame(records)


# class MaterialInference:
#     """
#     Infer material flows based on archetype rules.
#     Stores:
#       - value: inferred mean (deterministic)
#       - parameter_distribution: uncertainty model string
#       - value_formula: symbolic formula
#     """
#
#     def __init__(self, engine, material_rules_df):
#         self.engine = engine
#         self.material_rules_df = material_rules_df.copy()
#
#         required_cols = {"archetype", "material_name", "unit", "value", "reference_flow"}
#
#         missing = required_cols - set(self.material_rules_df.columns)
#         if missing:
#             raise ValueError(f"Material rules table missing columns: {missing}")
#
#         # Optional columns: make sure they exist
#         if "parameter_distribution" not in self.material_rules_df.columns:
#             self.material_rules_df["parameter_distribution"] = pd.NA
#
#         if "flow_type" not in self.material_rules_df.columns:
#             self.material_rules_df["flow_type"] = "Material"
#
#         if "source" not in self.material_rules_df.columns:
#             self.material_rules_df["source"] = ""
#
#     def infer_for_sites(self, site_ids, overwrite=False, verbose=True):
#         records = []
#
#         existing_pairs = set(zip(
#             self.engine.material_df.get("site_id", pd.Series(dtype=object)),
#             self.engine.material_df.get("subflow_type", pd.Series(dtype=object)),
#         ))
#
#         for site_id in site_ids:
#             site_row = self.engine.site_df[self.engine.site_df["site_id"] == site_id]
#             if site_row.empty:
#                 if verbose:
#                     print(f"⚠️ No site metadata for {site_id}")
#                 continue
#
#             meta = site_row.iloc[0]
#             archetype = meta.get("archetypes")
#
#             rules = self.material_rules_df[self.material_rules_df["archetype"] == archetype]
#             if rules.empty:
#                 if verbose:
#                     print(f"⚠️ No material rules for archetype '{archetype}'")
#                 continue
#
#             # Get production reference
#             prod_row = self.engine.production_df[self.engine.production_df["site_id"] == site_id]
#
#             for _, rule in rules.iterrows():
#                 subflow = rule["material_name"]
#                 if not overwrite and (site_id, subflow) in existing_pairs:
#                     continue
#
#                 ref_flow = rule["reference_flow"]  # e.g. "ore_processed_t"
#                 intensity_mean = rule["value"]  # e.g. 0.05 (kg/t)
#                 dist = rule.get("parameter_distribution")
#
#                 # Resolve reference flow value
#                 ref_value = None
#                 if not prod_row.empty and ref_flow in prod_row.columns:
#                     ref_value = prod_row.iloc[0][ref_flow]
#
#                 # ---- inferred deterministic value (mean) ----
#                 value_inferred = None
#                 if ref_value is not None and pd.notna(ref_value):
#                     value_inferred = ref_value * intensity_mean
#
#                 # ---- symbolic formula ----
#                 if pd.notna(dist):
#                     value_formula = f"{ref_flow} * {dist}"
#                 else:
#                     value_formula = f"{ref_flow} * {intensity_mean}"
#
#                 records.append({
#                     "site_id": site_id,
#                     "activity_name": meta.get("activity_name"),
#                     "mining_processing_type": meta.get("mining_processing_type"),
#                     "archetypes": archetype,
#                     "flow_type": rule["flow_type"],
#                     "subflow_type": subflow,
#                     "value": value_inferred,  # ✅ mean inferred value
#                     "unit": rule["unit"],
#                     "data_source": f"Archetype inference | {rule.get('source', '')}".strip(),
#                     "value_formula": value_formula,
#                     "amount_parameter": ref_flow,
#                     "parameter_distribution": dist,
#                 })
#
#         return pd.DataFrame(records, columns=INFERRED_COLS)

class MaterialInference:
    """
    Infer material use from archetype-based intensity rules.

    - value            : inferred mean (numeric)
    - value_formula    : symbolic scaling expression
    - parameter_distribution : uncertainty on intensity (optional)
    """

    def __init__(self, engine, material_rules_df):
        self.engine = engine
        self.material_rules_df = material_rules_df.copy()

        required_cols = {
            "archetype",
            "material_name",
            "value",   # mean intensity (e.g. kg / t ore)
            "unit",
        }
        missing = required_cols - set(self.material_rules_df.columns)
        if missing:
            raise ValueError(f"Material rules table missing columns: {missing}")

        # optional columns
        if "parameter_distribution" not in self.material_rules_df.columns:
            self.material_rules_df["parameter_distribution"] = pd.NA
        if "flow_type" not in self.material_rules_df.columns:
            self.material_rules_df["flow_type"] = "Material"
        if "source" not in self.material_rules_df.columns:
            self.material_rules_df["source"] = ""

    def infer_for_sites(self, site_ids, overwrite=False, verbose=True):
        records = []

        existing_pairs = set(zip(
            self.engine.material_df.get("site_id", pd.Series(dtype=object)),
            self.engine.material_df.get("subflow_type", pd.Series(dtype=object)),
        ))

        for site_id in site_ids:

            # ---- site metadata ----
            site_row = self.engine.site_df[self.engine.site_df["site_id"] == site_id]
            if site_row.empty:
                if verbose:
                    print(f"⚠️ No site metadata for {site_id}")
                continue
            meta = site_row.iloc[0]

            archetype = meta.get("archetypes")
            if not archetype or pd.isna(archetype):
                if verbose:
                    print(f"⚠️ No archetype for site {site_id}")
                continue

            # ---- production (FETCH ONCE) ----
            prod_row = self.engine.production_df[self.engine.production_df["site_id"] == site_id]
            ore_processed = None
            if not prod_row.empty and "ore_processed_t" in prod_row.columns:
                ore_processed = prod_row.iloc[0]["ore_processed_t"]

            if ore_processed is None or pd.isna(ore_processed):
                if verbose:
                    print(f"⚠️ No ore_processed_t for site {site_id}")

            # ---- rules ----
            rules = self.material_rules_df[self.material_rules_df["archetype"] == archetype]
            if rules.empty:
                if verbose:
                    print(f"⚠️ No material rules for archetype '{archetype}'")
                continue

            for _, rule in rules.iterrows():
                subflow = rule["material_name"]

                if not overwrite and (site_id, subflow) in existing_pairs:
                    continue

                intensity_mean = rule["value"]  # kg / t ore
                dist = rule["parameter_distribution"]

                # ---- inferred numeric value ----
                value_inferred = None
                if ore_processed is not None and pd.notna(ore_processed):
                    value_inferred = ore_processed * intensity_mean

                # ---- symbolic formula ----
                value_formula = f"ore_processed_t * {intensity_mean}"

                records.append({
                    "site_id": site_id,
                    "activity_name": meta.get("activity_name"),
                    "mining_processing_type": meta.get("mining_processing_type"),
                    "archetypes": archetype,
                    "flow_type": rule["flow_type"],
                    "subflow_type": subflow,
                    "value": value_inferred,                 # ✅ numeric mean
                    "unit": rule["unit"],
                    "data_source": f"Archetype inference | {rule['source']}".strip(),
                    "value_formula": value_formula,          # ✅ explicit
                    "parameter_distribution": dist,          # ✅ uncertainty
                })

        return pd.DataFrame(records, columns=INFERRED_COLS)



# class ExplosivesInference:
#     """
#     Infer explosive usage based on mining_processing_type.
#     Returns symbolic formulas and computes min, mean, max values where possible.
#     """
#
#     def __init__(self, engine):
#         self.engine = engine
#
#     @staticmethod
#     def _is_open_pit(mtype: Optional[str]) -> bool:
#         if not isinstance(mtype, str):
#             return False
#         return "open-pit" in mtype.lower() or "open pit" in mtype.lower()
#
#     @staticmethod
#     def _is_underground(mtype: Optional[str]) -> bool:
#         if not isinstance(mtype, str):
#             return False
#         return "underground" in mtype.lower()
#
#     @staticmethod
#     def _compute_explosives_values(ore_processed, sr_min=None, sr_max=None, ef_min=None, ef_max=None):
#         if ore_processed is None or np.isnan(ore_processed):
#             return None, None, None
#
#         if sr_min is not None:
#             min_val = ore_processed * sr_min * ef_min
#             mean_val = ore_processed * ((sr_min + sr_max) / 2) * ((ef_min + ef_max) / 2)
#             max_val = ore_processed * sr_max * ef_max
#         else:
#             min_val = ore_processed * ef_min
#             mean_val = ore_processed * ((ef_min + ef_max) / 2)
#             max_val = ore_processed * ef_max
#
#         return min_val, mean_val, max_val
#
#     def infer_for_sites(
#         self,
#         site_ids,
#         explosive_params: Dict[str, Dict[str, Tuple[float, float]]],
#         overwrite=False,
#         verbose=True,
#     ):
#         records = []
#         existing_pairs = set(
#             zip(
#                 self.engine.material_df["site_id"],
#                 self.engine.material_df["subflow_type"],
#             )
#         )
#
#         for site_id in site_ids:
#             site_row = self.engine.site_df[self.engine.site_df["site_id"] == site_id]
#             if site_row.empty:
#                 if verbose:
#                     print(f"⚠️ No site metadata for {site_id}")
#                 continue
#
#             site_meta = site_row.iloc[0]
#             mtype = site_meta.get("mining_processing_type")
#             archetype = site_meta.get("archetypes")
#             activity_name = site_meta.get("activity_name")
#
#             is_op = self._is_open_pit(mtype)
#             is_ug = self._is_underground(mtype)
#
#             if not (is_op or is_ug):
#                 continue
#
#             subflow = "Explosives"
#             if not overwrite and (site_id, subflow) in existing_pairs:
#                 continue
#
#             ore_row = self.engine.production_df[self.engine.production_df["site_id"] == site_id]
#             ore_processed = None
#             if not ore_row.empty:
#                 ore_processed = ore_row.iloc[0].get("ore_processed_t")
#
#             if is_op:
#                 sr_min, sr_max = explosive_params["open_pit"]["strip_ratio"]
#                 ef_min, ef_max = explosive_params["open_pit"]["explosive_factor"]
#                 value_formula = f"ore_processed_t * Uniform({sr_min}, {sr_max}) * Uniform({ef_min}, {ef_max})"
#                 val_min, val_mean, val_max = self._compute_explosives_values(ore_processed, sr_min, sr_max, ef_min, ef_max)
#                 param_dist = f"Uniform({sr_min}, {sr_max}) * Uniform({ef_min}, {ef_max})"
#                 source = "Inference (open-pit explosives)"
#             elif is_ug:
#                 ef_min, ef_max = explosive_params["underground"]["explosive_factor"]
#                 value_formula = f"ore_processed_t * Uniform({ef_min}, {ef_max})"
#                 val_min, val_mean, val_max = self._compute_explosives_values(ore_processed, None, None, ef_min, ef_max)
#                 param_dist = f"Uniform({ef_min}, {ef_max})"
#                 source = "Inference (underground explosives)"
#
#             rec = {
#                 "site_id": site_id,
#                 "activity_name": activity_name,
#                 "mining_processing_type": mtype,
#                 "archetypes": archetype,
#                 "flow_type": "Material",
#                 "subflow_type": subflow,
#                 "value": None,
#                 "unit": "kg",
#                 "data_source": source,
#                 "value_formula": value_formula,
#                 "amount_parameter": None,
#                 "parameter_distribution": param_dist,
#                 "value_min": val_min,
#                 "value_mean": val_mean,
#                 "value_max": val_max,
#                 #"reference_flow": "ore_processed_t"
#             }
#
#             records.append(rec)
#
#         cols = [
#             "site_id", "activity_name", "mining_processing_type", "archetypes",
#             "flow_type", "subflow_type", "value", "unit", "data_source", "value_formula",
#             "amount_parameter", "parameter_distribution", "value_min", "value_mean", "value_max"
#         ]
#
#         return pd.DataFrame(records, columns=cols)
class ExplosivesInference:

    def __init__(self, engine):
        self.engine = engine

    @staticmethod
    def _is_open_pit(mtype):
        return isinstance(mtype, str) and ("open-pit" in mtype.lower() or "open pit" in mtype.lower())

    @staticmethod
    def _is_underground(mtype):
        return isinstance(mtype, str) and ("underground" in mtype.lower())

    def infer_for_sites(self, site_ids, explosive_params, overwrite=False, verbose=True):
        records = []
        existing_pairs = set(zip(
            self.engine.material_df.get("site_id", pd.Series(dtype=object)),
            self.engine.material_df.get("subflow_type", pd.Series(dtype=object)),
        ))

        for site_id in site_ids:
            site_row = self.engine.site_df[self.engine.site_df["site_id"] == site_id]
            if site_row.empty:
                if verbose:
                    print(f"⚠️ No site metadata for {site_id}")
                continue

            meta = site_row.iloc[0]
            mtype = meta.get("mining_processing_type")

            prod_row = self.engine.production_df[self.engine.production_df["site_id"] == site_id]
            ore = prod_row.iloc[0].get("ore_processed_t") if not prod_row.empty else None

            subflow = "Explosives"
            if not overwrite and (site_id, subflow) in existing_pairs:
                continue

            if self._is_open_pit(mtype):
                sr_min, sr_max = explosive_params["open_pit"]["strip_ratio"]
                ef_min, ef_max = explosive_params["open_pit"]["explosive_factor"]
                dist = f"Uniform({sr_min}, {sr_max}) * Uniform({ef_min}, {ef_max})"
                formula = f"ore_processed_t * {dist}"
                source = "Inference | explosives | open-pit"

            elif self._is_underground(mtype):
                ef_min, ef_max = explosive_params["underground"]["explosive_factor"]
                dist = f"Uniform({ef_min}, {ef_max})"
                formula = f"ore_processed_t * {dist}"
                source = "Inference | explosives | underground"
            else:
                continue

            mean_factor = mean_of_distribution(dist)
            value_inferred = None
            if mean_factor is not None and ore is not None and pd.notna(ore):
                value_inferred = ore * mean_factor

            records.append({
                "site_id": site_id,
                "activity_name": meta.get("activity_name"),
                "mining_processing_type": mtype,
                "archetypes": meta.get("archetypes"),
                "flow_type": "Material",
                "subflow_type": subflow,
                "value": value_inferred,  # ✅ inferred mean
                "unit": "kg",
                "data_source": source,
                "value_formula": formula,
                "amount_parameter": "ore_processed_t",
                "parameter_distribution": dist,
            })

        return pd.DataFrame(records, columns=INFERRED_COLS)


class CementInference:
    """
    Infer cement for backfilling.

    Logic identical to MaterialInference, but without archetype rules:
    - value = ore_processed_t * mean(cement intensity)
    - value_formula = ore_processed_t * <intensity>
    - parameter_distribution = uncertainty on intensity
    """

    def __init__(self, engine):
        self.engine = engine

    @staticmethod
    def _get_method_key(mtype):
        if not isinstance(mtype, str):
            return None
        m = mtype.lower()
        if "underground" in m:
            return "underground"
        if "open-pit" in m or "open pit" in m:
            return "open_pit"
        return None

    def infer_for_sites(self, site_ids, cement_params, overwrite=False, verbose=True):
        records = []

        existing_pairs = set(zip(
            self.engine.material_df.get("site_id", pd.Series(dtype=object)),
            self.engine.material_df.get("subflow_type", pd.Series(dtype=object)),
        ))

        for site_id in site_ids:

            # ---- site metadata ----
            site_row = self.engine.site_df[self.engine.site_df["site_id"] == site_id]
            if site_row.empty:
                if verbose:
                    print(f"⚠️ No site metadata for {site_id}")
                continue

            meta = site_row.iloc[0]
            mtype = meta.get("mining_processing_type")
            method_key = self._get_method_key(mtype)

            # ---- no cement defined for this mining method ----
            params = cement_params.get(method_key)
            if params is None:
                continue

            if "cement_factor" not in params:
                continue

            cmin, cmax = params["cement_factor"]
            dist = f"Uniform({cmin}, {cmax})"

            # ---- production ----
            prod_row = self.engine.production_df[self.engine.production_df["site_id"] == site_id]
            ore = None
            if not prod_row.empty and "ore_processed_t" in prod_row.columns:
                ore = prod_row.iloc[0]["ore_processed_t"]

            subflow = "Cement (backfilling)"
            if not overwrite and (site_id, subflow) in existing_pairs:
                continue

            # ---- inferred mean ----
            value_inferred = None
            mean_factor = mean_of_distribution(dist)
            if mean_factor is not None and ore is not None and pd.notna(ore):
                value_inferred = ore * mean_factor

            records.append({
                "site_id": site_id,
                "activity_name": meta.get("activity_name"),
                "mining_processing_type": mtype,
                "archetypes": meta.get("archetypes"),
                "flow_type": "Material",
                "subflow_type": subflow,
                "value": value_inferred,                # ✅ numeric mean
                "unit": "kg",
                "data_source": f"Inference | Cement backfilling | {method_key}",
                "value_formula": f"ore_processed_t * {dist}",
                "parameter_distribution": dist,
            })

        return pd.DataFrame(records, columns=INFERRED_COLS)

# class CementInference:
#     """
#     Infer cement use for mine backfilling based on mining method.
#     Now returns Brightway-ready symbolic + min/mean/max estimates.
#     """
#
#     def __init__(self, engine):
#         self.engine = engine
#
#     @staticmethod
#     def _get_method_key(mtype):
#         if not isinstance(mtype, str):
#             return None
#         m = mtype.lower()
#         if "underground" in m:
#             return "underground"
#         if "open-pit" in m or "open pit" in m:
#             return "open_pit"
#         return None
#
#     def infer_for_sites(
#         self,
#         site_ids,
#         cement_params,
#         overwrite=False,
#         verbose=True,
#     ):
#         records = []
#         existing_pairs = set(
#             zip(self.engine.material_df["site_id"], self.engine.material_df["subflow_type"])
#         )
#
#         for site_id in site_ids:
#             site_row = self.engine.site_df[self.engine.site_df["site_id"] == site_id]
#             if site_row.empty:
#                 if verbose:
#                     print(f"⚠️ No site metadata for {site_id}")
#                 continue
#
#             meta = site_row.iloc[0]
#             mtype = meta.get("mining_processing_type")
#             archetype = meta.get("archetypes")
#             activity_name = meta.get("activity_name")
#
#             method_key = self._get_method_key(mtype)
#             if method_key not in cement_params:
#                 continue
#
#             params = cement_params.get(method_key)
#             if not params or "cement_factor" not in params:
#                 continue
#
#             cmin, cmax = params["cement_factor"]
#             ore_row = self.engine.production_df[self.engine.production_df["site_id"] == site_id]
#             ore = ore_row.iloc[0]["ore_processed_t"] if not ore_row.empty else None
#
#             subflow = "Cement (backfilling)"
#             if not overwrite and (site_id, subflow) in existing_pairs:
#                 continue
#
#             value_formula = f"ore_processed_t * Uniform({cmin}, {cmax})"
#             value_min = value_mean = value_max = None
#
#             if ore is not None and pd.notna(ore):
#                 value_min = ore * cmin
#                 value_max = ore * cmax
#                 value_mean = ore * (cmin + cmax) / 2
#
#             records.append({
#                 "site_id": site_id,
#                 "activity_name": activity_name,
#                 "mining_processing_type": mtype,
#                 "archetypes": archetype,
#                 "flow_type": "Material",
#                 "subflow_type": subflow,
#                 "value": None,
#                 "unit": "kg",
#                 #"reference_flow": "ore_processed_t",
#                 "data_source": (
#                     f"Inference | Cement backfilling | {method_key} | {cmin}-{cmax} kg/t ore"
#                 ),
#                 "value_formula": value_formula,
#                 "amount_parameter": "ore_processed_t",
#                 "parameter_distribution": f"Uniform({cmin}, {cmax})",
#                 "value_min": value_min,
#                 "value_mean": value_mean,
#                 "value_max": value_max,
#             })
#
#         columns = [
#             "site_id", "activity_name", "mining_processing_type", "archetypes",
#             "flow_type", "subflow_type", "value", "unit",
#             "data_source", "value_formula", "amount_parameter",
#             "parameter_distribution", "value_min", "value_mean", "value_max"
#         ]
#         return pd.DataFrame(records, columns=columns)


