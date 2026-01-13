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
    electricity_inference: "ElectricityInference" = field(init=False)
    land_inference: "LandInference" = field(init=False)
    material_inference: "MaterialInference" = field(init=False)
    explosives_inference: "ExplosivesInference" = field(init=False)
    cement_inference: "CementInference" = field(init=False)

    def __post_init__(self):
        self.energy_inference = EnergyInference(self)
        self.electricity_inference = ElectricityInference(self)
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

    def infer_electricity_for_sites(
            self,
            site_ids,
            electricity_share_rules,
            overwrite=False,
            verbose=True,
    ):
        inferred = self.electricity_inference.infer_for_sites(
            site_ids=site_ids,
            electricity_share_rules=electricity_share_rules,
            overwrite=overwrite,
            verbose=verbose,
        )

        self.energy_df = pd.concat(
            [
                self.energy_df.reindex(columns=INFERRED_COLS),
                inferred.reindex(columns=INFERRED_COLS),
            ],
            ignore_index=True,
        )

        return self.energy_df, inferred


    def infer_land_for_sites(
            self,
            site_ids,
            formula_open_pit=None,
            underground_land_factors=None,
            formula_other="Uniform(other_land_min, other_land_max)",
            overwrite=False
    ):
        inferred = self.land_inference.infer_for_sites(
            site_ids=site_ids,
            formula_open_pit=formula_open_pit,
            underground_land_factors=underground_land_factors,
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


class ElectricityInference:
    """
    Infer electricity consumption as a share of total final energy (MJ).

    - Converts all fuel energy to MJ
    - Applies electricity share by mining method / archetype
    - Adds one 'Electricity' energy flow (MJ)
    """

    def __init__(self, engine):
        self.engine = engine

    @staticmethod
    def _get_method_key(mtype):
        if not isinstance(mtype, str):
            return None

        m = mtype.lower()
        has_op = ("open-pit" in m) or ("open pit" in m)
        has_ug = ("underground" in m)

        if has_op and has_ug:
            return "mixed"
        if has_ug:
            return "underground"
        if has_op:
            return "open-pit"
        return None

    def infer_for_sites(
        self,
        site_ids,
        electricity_share_rules,
        overwrite=False,
        verbose=True,
    ):

        ENERGY_TO_MJ = {
            "Diesel": 38.6,  # MJ / L
            "LPG": 25.3,  # MJ / L assumed same as propane
            "Natural gas": 38.0,  # MJ / m3
        }

        records = []

        existing_pairs = set(zip(
            self.engine.energy_df.get("site_id", pd.Series(dtype=object)),
            self.engine.energy_df.get("subflow_type", pd.Series(dtype=object)),
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
            archetype = meta.get("archetypes")

            subflow = "Electricity consumption|Grid electricity"
            if not overwrite and (site_id, subflow) in existing_pairs:
                continue

            # ---- select electricity share ----
            method_key = self._get_method_key(mtype)
            if method_key not in electricity_share_rules:
                if verbose:
                    print(f"⚠️ No electricity share rule for {method_key}")
                continue

            rules = electricity_share_rules[method_key]
            share = rules.get(archetype, rules.get("default"))

            if share is None:
                continue

            if isinstance(share, (int, float)):
                smin = smax = float(share)
            else:
                smin, smax = share

            mean_share = (smin + smax) / 2

            # ---- collect site energy flows ----
            df_energy = self.engine.energy_df[self.engine.energy_df["site_id"] == site_id]
            if df_energy.empty:
                continue

            total_energy_MJ = 0.0

            for _, r in df_energy.iterrows():
                sub = r["subflow_type"]
                val = r["value"]
                unit = r["unit"]

                if val is None or pd.isna(val):
                    continue

                # identify fuel
                if sub.startswith("Diesel"):
                    total_energy_MJ += val * ENERGY_TO_MJ["Diesel"]
                elif sub.startswith("LPG"):
                    total_energy_MJ += val * ENERGY_TO_MJ["LPG"]
                elif sub.startswith("Natural gas"):
                    total_energy_MJ += val * ENERGY_TO_MJ["Natural gas"]
                elif sub == "Electricity":
                    # already electricity → skip
                    continue

            if total_energy_MJ == 0:
                continue

            # ---- infer electricity ----
            fuel_energy_MJ = total_energy_MJ
            electricity_MJ = (mean_share / (1 - mean_share)) * fuel_energy_MJ
            dist = f"Uniform({smin}, {smax})" if smin != smax else None
            formula = f"({mean_share} / (1 - {mean_share})) * fuel_energy_MJ"


            records.append({
                "site_id": site_id,
                "activity_name": meta.get("activity_name"),
                "mining_processing_type": mtype,
                "archetypes": archetype,
                "flow_type": "Energy",
                "subflow_type": "Electricity consumption|Grid electricity",
                "value": electricity_MJ,
                "unit": "MJ",
                "data_source": "Inference | electricity share of total energy",
                "value_formula": formula,
                "parameter_distribution": dist,
            })

        return pd.DataFrame(records, columns=INFERRED_COLS)


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
        underground_land_factors=None,
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
            archetype = meta.get("archetypes")

            prod_row = self.engine.production_df[self.engine.production_df["site_id"] == site_id]
            ore = prod_row.iloc[0].get("ore_processed_t") if not prod_row.empty else None

            subflow = "Land area"
            if not overwrite and (site_id, subflow) in existing_pairs:
                continue

            value_inferred = None
            dist = None

            if self._is_open_pit(mtype):
                formula = formula_open_pit
                source = "Inference | land | open-pit regression"
                if ore is not None and pd.notna(ore):
                    value_inferred = float(
                        eval(formula, {"__builtins__": {}}, {"ore_processed_t": ore})
                    )

            elif self._is_underground(mtype) and underground_land_factors:

                params = underground_land_factors.get(archetype)
                if params is None:
                    continue

                factor = params["factor"]

                # ---- normalize factor (scalar or range) ----
                if isinstance(factor, (int, float)):
                    xmin = xmax = float(factor)
                elif isinstance(factor, (tuple, list)) and len(factor) == 2:
                    xmin, xmax = factor
                else:
                    raise ValueError(
                        f"Invalid land factor for archetype '{archetype}': {factor}"
                    )

                dist = f"Uniform({xmin}, {xmax})" if xmin != xmax else None
                formula = (
                    f"ore_processed_t * {xmin}"
                    if xmin == xmax
                    else f"ore_processed_t * Uniform({xmin}, {xmax})"
                )
                source = f"Inference | land | underground | {archetype}"

                mean_factor = (xmin + xmax) / 2
                if ore is not None and pd.notna(ore):
                    value_inferred = ore * mean_factor


            else:
                dist = formula_other
                formula = f"ore_processed_t * {dist}"
                source = "Inference | land | other facilities"

                mean_factor = mean_of_distribution(dist)
                if mean_factor is not None and ore is not None and pd.notna(ore):
                    value_inferred = ore * mean_factor

            records.append({
                "site_id": site_id,
                "activity_name": meta.get("activity_name"),
                "mining_processing_type": mtype,
                "archetypes": archetype,
                "flow_type": "Land",
                "subflow_type": subflow,
                "value": value_inferred,
                "unit": "m2",
                "data_source": source,
                "value_formula": formula,
                "parameter_distribution": dist,
            })

        return pd.DataFrame(records, columns=INFERRED_COLS)


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


