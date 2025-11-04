import pandas as pd
import numpy as np
import re
from utils.constants import *


# ======================================================
def ore_to_concentrate(
    ore_processed_tonnes: float,
    head_grade: float,
    recovery_rate: float,
    concentrate_grade: float
):
    """
    Compute concentrate mass and related values.
    All grades and recovery rates are fractions (e.g. 0.005 = 0.5%).
    """
    metal_in_ore = ore_processed_tonnes * head_grade
    metal_in_concentrate = metal_in_ore * recovery_rate
    concentrate_mass = metal_in_concentrate / concentrate_grade
    yield_fraction = concentrate_mass / ore_processed_tonnes
    tailings_grade = head_grade * (1 - recovery_rate)

    return {
        "M_concentrate_t": concentrate_mass,
        "M_metal_in_ore_t": metal_in_ore,
        "M_metal_in_concentrate_t": metal_in_concentrate,
        "Yield_fraction": yield_fraction,
        "Tailings_grade": tailings_grade
    }


# ======================================================
# Needed for normalization
# ======================================================
def _norm_unit(x):
    if pd.isna(x): return None
    return str(x).strip().lower().replace(' ', '')

def _canon_subflow(x):
    """
    Canonicalize subflow names for consistent matching with constants.py dictionaries.
    - Keeps spaces and parentheses (important for acids and detailed names).
    - Lowercases everything.
    - Removes text after '|' (if present).
    - Applies SUBFLOW_ALIASES if a match is found.
    """
    if pd.isna(x):
        return None
    s = str(x).strip().lower()
    if '|' in s:
        s = s.split('|', 1)[0].strip()
    return SUBFLOW_ALIASES.get(s, s)


def standardize_materials_to_t(df, subflow_col='subflow_type', unit_col='unit', value_col='value',
                               density_table=None):
    """
    Convert 'material' rows to tonnes (t).
    Handles kg, t, L (via density), and pounds.
    """
    den = {k.lower(): v for k, v in (density_table or DEFAULT_DENSITY).items()}
    out = df.copy()

    out['_unit_n'] = out[unit_col].astype(str).str.strip().str.lower().str.replace(' ', '', regex=False)
    out['_subflow_n'] = out[subflow_col].map(_canon_subflow)
    out[value_col] = pd.to_numeric(out[value_col], errors='coerce')

    # --- Direct tonnes ---
    mask_t = out['_unit_n'].isin({'t','tonne','tonnes','metricton','ton'})
    out.loc[mask_t, 'mass_t'] = out.loc[mask_t, value_col]
    out.loc[mask_t, 'mass_source'] = 't'
    out.loc[mask_t, 'mass_note'] = 'reported in tonnes'

    # --- kg → t ---
    mask_kg = out['_unit_n'].isin({'kg','kilogram','kilograms'})
    out.loc[mask_kg, 'mass_t'] = out.loc[mask_kg, value_col] / 1000.0
    out.loc[mask_kg, 'mass_source'] = 'kg→t'
    out.loc[mask_kg, 'mass_note'] = 'kg/1000'

    # --- pounds → t ---
    mask_lb = out['_unit_n'].isin({'lb', 'lbs', 'pound', 'pounds'})
    if mask_lb.any():
        out.loc[mask_lb, 'mass_t'] = out.loc[mask_lb, value_col] * 0.453592 / 1000.0
        out.loc[mask_lb, 'mass_source'] = 'lbs→t'
        out.loc[mask_lb, 'mass_note'] = 'lbs×0.453592/1000'

    # --- liters → t via density ---
    mask_L = out['_unit_n'].isin(VOLUME_TO_L)
    if mask_L.any():
        multL = out.loc[mask_L, '_unit_n'].map(VOLUME_TO_L)
        dens = out.loc[mask_L, '_subflow_n'].map(lambda s: den.get(s if s else '', np.nan))
        mass_t = (out.loc[mask_L, value_col] * multL * dens) / 1000.0
        out.loc[mask_L, 'mass_t'] = mass_t
        out.loc[mask_L, 'mass_source'] = np.where(dens.notna(), 'L×density→t', 'missing_density')
        out.loc[mask_L, 'mass_note'] = np.where(
            dens.notna(),
            out.loc[mask_L, '_unit_n'] + "×density=" + dens.map(lambda x: f"{x:g}") + " kg/L",
            "volume reported; no density mapping for this subflow"
        )

    # --- Unknown ---
    mask_done = mask_t | mask_kg | mask_lb | mask_L
    out.loc[~mask_done & out[value_col].notna(), 'mass_source'] = 'unknown_unit'
    out.loc[~mask_done & out[value_col].notna(), 'mass_note'] = 'no rule for this unit'

    out['needs_density'] = (out['mass_source'] == 'missing_density')

    out = out.drop(columns=['_unit_n', '_subflow_n'])
    return out


def standardize_energy_to_MJ(
    df,
    subflow_col='subflow_type',
    unit_col='unit',
    value_col='value',
    lhv_table=None,
    equivalence_table=None
):
    """
    Convert energy/fuel rows to MJ (or equivalent MJ if non-fuel carrier like electricity).

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing energy-related flows.
    subflow_col, unit_col, value_col : str
        Column names for subflow, unit, and value.
    lhv_table : dict, optional
        Lower heating value (LHV) lookup table. Defaults to DEFAULT_LHV.
    equivalence_table : dict, optional
        Energy equivalence lookup (e.g., electricity→diesel). Defaults to ENERGY_EQUIVALENCE.
    """
    from utils.constants import (
        UNIT_TO_MJ, VOLUME_TO_L, CUBIC_M_TO_M3,
        DEFAULT_LHV, ENERGY_EQUIVALENCE
    )

    lhv = (lhv_table or DEFAULT_LHV).copy()
    eqv = (equivalence_table or ENERGY_EQUIVALENCE).copy()
    out = df.copy()

    # Normalize
    out['_unit_n'] = out[unit_col].map(_norm_unit)
    out['_subflow_n'] = out[subflow_col].map(_canon_subflow)
    out[value_col] = pd.to_numeric(out[value_col], errors='coerce')

    # 1️⃣ Direct energy units (MJ, kWh, etc.)
    direct_mask = out['_unit_n'].isin(UNIT_TO_MJ)
    out.loc[direct_mask, 'value_MJ'] = (
        out.loc[direct_mask, value_col] *
        out.loc[direct_mask, '_unit_n'].map(UNIT_TO_MJ)
    )
    out.loc[direct_mask, 'unit_source'] = 'direct_unit'
    out.loc[direct_mask, 'assumption_note'] = (
        out.loc[direct_mask, '_unit_n'].map(lambda u: f"{u}→MJ factor={UNIT_TO_MJ[u]}")
    )

    # 2️⃣ Fuels via LHV (mass or volume)
    fuel_rows = ~direct_mask & out['_subflow_n'].notna() & out[value_col].notna()
    for idx in out.index[fuel_rows]:
        sub = out.at[idx, '_subflow_n']
        unit = out.at[idx, '_unit_n']
        val  = out.at[idx, value_col]
        lhv_data = lhv.get(sub)
        converted = False

        if not lhv_data:
            out.at[idx, 'unit_source'] = 'missing_factor'
            out.at[idx, 'assumption_note'] = f"No LHV for subflow={sub}"
            continue

        # --- Mass-based fuels (kg, t)
        if unit in ('kg', 'kilogram', 't', 'tonne', 'tonnes'):
            factor = lhv_data.get('MJ/kg')
            if factor:
                mult = 1000.0 if unit.startswith('t') else 1.0
                out.at[idx, 'value_MJ'] = val * mult * factor
                out.at[idx, 'unit_source'] = 'lhv_factor'
                out.at[idx, 'assumption_note'] = f"{sub} MJ/kg={factor}"
                converted = True

        # --- Volume-based fuels (L, m3)
        elif unit in VOLUME_TO_L:
            mult_L = VOLUME_TO_L[unit]
            dens = lhv_data.get('density_kg_per_L')
            factor_kg = lhv_data.get('MJ/kg')
            if dens and factor_kg:
                out.at[idx, 'value_MJ'] = val * mult_L * dens * factor_kg
                out.at[idx, 'unit_source'] = 'lhv+density'
                out.at[idx, 'assumption_note'] = f"{sub} MJ/kg={factor_kg}, dens={dens} kg/L"
                converted = True
        elif unit in CUBIC_M_TO_M3:
            factor_m3 = lhv_data.get('MJ/m3')
            if factor_m3:
                out.at[idx, 'value_MJ'] = val * factor_m3
                out.at[idx, 'unit_source'] = 'lhv_factor'
                out.at[idx, 'assumption_note'] = f"{sub} MJ/m3={factor_m3}"
                converted = True

        # --- Electricity to fuel equivalence
        if not converted and sub == 'electricity':
            eq = eqv.get(('electricity', 'diesel'))
            if eq:
                eff = eq['efficiency']
                lhv_diesel = eq['MJ/kg']
                # Convert MJ electricity to kg diesel equivalent
                out.at[idx, 'value_MJ'] = val  # stays in MJ (already)
                out.at[idx, 'diesel_eq_kg'] = val / (eff * lhv_diesel)
                out.at[idx, 'unit_source'] = 'electricity_equiv'
                out.at[idx, 'assumption_note'] = f"electricity→diesel eq. eff={eff}, LHV={lhv_diesel}"

        if not converted and sub != 'electricity':
            out.at[idx, 'unit_source'] = 'missing_factor'
            out.at[idx, 'assumption_note'] = f"No conversion rule for {sub} [{unit}]"

    # Final flags
    out['unit_standard'] = np.where(out['value_MJ'].notna(), 'MJ', None)
    out['needs_factor'] = out['value_MJ'].isna() & out[value_col].notna()
    out = out.drop(columns=['_unit_n', '_subflow_n'])
    return out
#
#
# def standardize_materials_to_t(df, subflow_col='subflow_type', unit_col='unit', value_col='value',
#                                density_table=None):
#     """
#     Convert 'material' rows to tonnes.
#     Adds:
#       - mass_t : numeric mass in tonnes
#       - mass_source : 't','kg→t','L×density→t','missing_density','unknown_unit'
#       - mass_note : short note on the assumption used
#       - needs_density : True when a volume row had no density mapping
#     """
#     den = {k.lower(): v for k, v in (density_table or DEFAULT_DENSITY).items()}
#     out = df.copy()
#
#     out['_unit_n'] = out[unit_col].astype(str).str.strip().str.lower().str.replace(' ', '', regex=False)
#     out['_subflow_n'] = out[subflow_col].map(_canon_subflow)
#     out[value_col] = pd.to_numeric(out[value_col], errors='coerce')
#
#     # direct tonnes
#     mask_t = out['_unit_n'].isin({'t','tonne','tonnes','metricton','ton'})
#     out.loc[mask_t, 'mass_t'] = out.loc[mask_t, value_col].astype(float)
#     out.loc[mask_t, 'mass_source'] = 't'
#     out.loc[mask_t, 'mass_note'] = 'reported in tonnes'
#
#     # kg → t
#     mask_kg = out['_unit_n'].isin({'kg','kilogram','kilograms'})
#     out.loc[mask_kg, 'mass_t'] = out.loc[mask_kg, value_col] / 1000.0
#     out.loc[mask_kg, 'mass_source'] = 'kg→t'
#     out.loc[mask_kg, 'mass_note'] = 'kg/1000'
#
#     # liters family → t using density (kg/L)
#     mask_L = out['_unit_n'].isin(VOLUME_TO_L)
#     if mask_L.any():
#         multL = out.loc[mask_L, '_unit_n'].map(VOLUME_TO_L)
#         # find density per row from mapping on canonical subflow
#         dens = out.loc[mask_L, '_subflow_n'].map(lambda s: den.get(s if s else '', np.nan))
#         mass_t = (out.loc[mask_L, value_col] * multL * dens) / 1000.0
#         out.loc[mask_L, 'mass_t'] = mass_t
#         out.loc[mask_L, 'mass_source'] = np.where(dens.notna(), 'L×density→t', 'missing_density')
#         out.loc[mask_L, 'mass_note'] = np.where(
#             dens.notna(),
#             (out.loc[mask_L, '_unit_n'].map(str) + f"→L × density kg/L; density=" + dens.map(lambda x: f"{x:g}")),
#             "volume reported; no density mapping for this subflow"
#         )
#
#     # mark unknown units
#     mask_done = mask_t | mask_kg | mask_L
#     out.loc[~mask_done & out[value_col].notna(), 'mass_source'] = 'unknown_unit'
#     out.loc[~mask_done & out[value_col].notna(), 'mass_note'] = 'no rule for this unit'
#
#     out['needs_density'] = (out['mass_source'] == 'missing_density')
#
#     # clean temp
#     out = out.drop(columns=['_unit_n','_subflow_n'])
#     return out


# ======================================================
# Needed for conversion to ecoinvent
# ======================================================
def map_technosphere_to_ecoinvent(technosphere_df, mapping_df, ca_provinces_dict):
    """
    Map MetalliCan technosphere flows to Ecoinvent flows with consistent physical conversions.
    Extended handling for:
      - Electricity (on-site → diesel; grid → kWh)
      - LPG, oil, petrol, used oil, acetylene
      - Non-renewable fuel use, generic 'energy use'
      - ANFO/emulsion explosives
    """

    import pandas as pd

    merged_df = technosphere_df.merge(
        mapping_df,
        left_on="subflow_type",
        right_on="MetalliCan",
        how="left"
    )

    # --- Warn for unmapped flows ---
    unmapped = merged_df[merged_df["Flow name"].isna()]["subflow_type"].unique().tolist()
    if len(unmapped) > 0:
        print("⚠️ Les flux suivants n'ont pas trouvé de correspondance dans Ecoinvent:")
        for f in unmapped:
            print(f" - {f}")

    merged_df["Flow name"] = merged_df["Flow name"].fillna("No mapping")
    merged_df["Reference product"] = merged_df["Reference product"].fillna("No mapping")
    merged_df["Unit"] = merged_df["Unit"].fillna(merged_df["unit"])

    # --- Location correction ---
    merged_df["Location"] = merged_df.apply(
        lambda row: ca_provinces_dict.get(row["province"], row["Location"])
        if isinstance(row["Location"], str) and row["Location"].startswith("CA-")
        else row["Location"],
        axis=1
    )

    # --- Base conversion factors ---
    conversion_factors = {
        ("MJ", "MJ"): 1.0,
        ("t", "t"): 1.0,
        ("MJ", "kilowatt hour"): 0.277778,
        ("kilowatt hour", "MJ"): 3.6,
        ("kg", "kilogram"): 1.0,
        ("kilogram", "kg"): 1.0,
        ("t", "kilogram"): 1000.0,
        ("kilogram", "tonne"): 0.001,
    }

    # --- Physical constants (literature sources) ---
    fuel_lhv_mj_per_kg = {
        "diesel": 43.0,
        "gasoline": 44.4,
        "propane": 46.4,
        "kerosene": 43.1,
        "aviation fuel": 43.1,
        "biodiesel": 37.4,
        "naphtha": 44.9,
        "natural gas_mj_per_m3": 35.2,
        "lpg": 46.1,
        "petrol": 44.4,
        "oil": 42.7,
        "used oil": 40.0,
        "acetylene": 48.0,
    }

    density_kg_per_m3 = {
        "biodiesel": 877.0,
        "naphtha": 725.0,
    }

    explosives_energy_mj_per_kg = {
        "anfo": 2.3,
        "emulsion": 3.7,
        "explosive_default": 3.0
    }

    # --- Conversion function ---
    def convert_value(row):
        val = row["value_normalized"]
        if pd.isna(val):
            return None

        original_unit = str(row["unit"]).strip()
        target_unit = str(row["Unit"]).strip()
        subflow_lower = str(row["subflow_type"]).lower()
        ref_product = str(row.get("Reference product", "")).lower()
        flow_name = str(row.get("Flow name", "")).lower()

        # direct unit mapping
        key = (original_unit, target_unit)
        if key in conversion_factors:
            return val * conversion_factors[key]

        # --- Electricity cases ---
        if "electricity" in subflow_lower:
            # grid electricity (non-renewable electricity use)
            if "non-renewable" in subflow_lower or "grid" in flow_name:
                if original_unit == "MJ" and target_unit in ["kilowatt hour", "kwh"]:
                    return val * 0.277778
                if original_unit in ["kilowatt hour", "kwh"] and target_unit == "MJ":
                    return val * 3.6
                return val
            # on-site electricity mapped to diesel
            if "on-site" in subflow_lower or "generator" in flow_name or "diesel" in ref_product:
                lhv = fuel_lhv_mj_per_kg["diesel"]
                if original_unit == "MJ" and target_unit in ["kg", "kilogram"]:
                    return val / lhv
                if original_unit in ["kg", "kilogram"] and target_unit == "MJ":
                    return val * lhv

        # --- Generic energy placeholders (diesel equivalent) ---
        if any(k in subflow_lower for k in ["non-renewable fuel", "energy use", "fuel use"]):
            lhv = fuel_lhv_mj_per_kg["diesel"]
            if original_unit == "MJ" and target_unit in ["kg", "kilogram"]:
                return val / lhv
            if original_unit in ["kg", "kilogram"] and target_unit == "MJ":
                return val * lhv

        # --- Specific fuels ---
        for fuel, lhv in fuel_lhv_mj_per_kg.items():
            if fuel.endswith("_mj_per_m3"):
                continue
            if fuel in subflow_lower:
                if original_unit == "MJ" and target_unit in ["kg", "kilogram"]:
                    return val / lhv
                if original_unit in ["kg", "kilogram"] and target_unit == "MJ":
                    return val * lhv

        # --- Volume-based fuels ---
        if "biodiesel" in subflow_lower:
            lhv = fuel_lhv_mj_per_kg["biodiesel"]
            dens = density_kg_per_m3["biodiesel"]
            mj_per_m3 = lhv * dens
            if original_unit == "MJ" and target_unit in ["cubic meter", "m3"]:
                return val / mj_per_m3
            if original_unit in ["cubic meter", "m3"] and target_unit == "MJ":
                return val * mj_per_m3

        if "natural gas" in subflow_lower:
            mj_per_m3 = fuel_lhv_mj_per_kg["natural gas_mj_per_m3"]
            if original_unit == "MJ" and target_unit in ["cubic meter", "m3"]:
                return val / mj_per_m3
            if original_unit in ["cubic meter", "m3"] and target_unit == "MJ":
                return val * mj_per_m3

        if "naphtha" in subflow_lower or "naphta" in subflow_lower:
            lhv = fuel_lhv_mj_per_kg["naphtha"]
            dens = density_kg_per_m3["naphtha"]
            mj_per_m3 = lhv * dens
            if original_unit == "MJ" and target_unit in ["kg", "kilogram"]:
                return val / lhv
            if original_unit in ["kg", "kilogram"] and target_unit == "MJ":
                return val * lhv
            if original_unit == "MJ" and target_unit in ["cubic meter", "m3"]:
                return val / mj_per_m3
            if original_unit in ["cubic meter", "m3"] and target_unit == "MJ":
                return val * mj_per_m3

        # --- Explosives (ANFO, emulsion, generic) ---
        if "anfo" in subflow_lower or "emulsion" in subflow_lower:
            e = explosives_energy_mj_per_kg["emulsion" if "emulsion" in subflow_lower else "anfo"]
            if original_unit == "MJ" and target_unit in ["kg", "kilogram"]:
                return val / e
            if original_unit in ["kg", "kilogram"] and target_unit == "MJ":
                return val * e
        if "explos" in subflow_lower:
            e = explosives_energy_mj_per_kg["explosive_default"]
            if original_unit == "MJ" and target_unit in ["kg", "kilogram"]:
                return val / e
            if original_unit in ["kg", "kilogram"] and target_unit == "MJ":
                return val * e

        # --- Fallback warning ---
        print(f"⚠️ Pas de conversion définie pour {original_unit} → {target_unit} (flux: {row['subflow_type']})")
        return val

    merged_df["value_converted"] = merged_df.apply(convert_value, axis=1)

    # --- Final tidy dataframe ---
    result_df = merged_df[[
        "activity_name", "functional_unit", "site_id", "subflow_type",
        "Flow name", "Reference product", "value_normalized", "unit",
        "value_converted", "Unit", "Location", "province", "DB_to_map"
    ]].rename(columns={
        "Flow name": "Activity",
        "Reference product": "Product",
        #"unit": "unit_original",
        "value_converted": "Amount",
        #"Unit": "unit_target",
        "DB_to_map": "Database"
    })

    return result_df


def map_biosphere_to_ecoinvent(biosphere_df, mapping_df, ca_provinces_dict):
    """
    Map site-level biosphere flows to Ecoinvent biosphere flows.

    - Merges with a biosphere mapping file (MetalliCan → Ecoinvent).
    - Warns for unmapped flows.
    - Converts units when possible.
    - Updates flow names with provincial location suffixes (CA-XX) when needed.
    """

    # --- 1️⃣ Merge with mapping
    merged_df = biosphere_df.merge(
        mapping_df,
        left_on="substance_name",
        right_on="substance_name",
        how="left"
    )

    print(merged_df.columns)

    # --- 2️⃣ Warn for unmapped flows
    unmapped = merged_df[merged_df["Flow name"].isna()]["substance_name"].unique().tolist()
    if len(unmapped) > 0:
        print(f"⚠️ {len(unmapped)} biosphere flows could not be mapped to Ecoinvent:")
        for f in unmapped:
            print(f"   - {f}")

    # --- 3️⃣ Fill missing mappings
    merged_df["Flow name"] = merged_df["Flow name"].fillna("No mapping")
    merged_df["Compartments"] = merged_df["Compartments"].fillna("No mapping")
    #merged_df["Unit"] = merged_df["Unit"].fillna(merged_df["unit"])
    merged_df["DB_to_map"] = merged_df["DB_to_map"].fillna("biosphere3")

    # --- 4️⃣ Replace "CA-xx" placeholders with correct province codes
    def fix_ca_placeholder(row):
        flow = str(row["Flow name"])
        province = row.get("province", None)
        # Only modify if flow name contains "CA-xx"
        if re.search(r"CA-xx$", flow, flags=re.IGNORECASE) and province in ca_provinces_dict:
            correct_suffix = ca_provinces_dict[province]
            # Replace "CA-xx" (case insensitive) by correct code
            return re.sub(r"CA-xx$", correct_suffix, flow, flags=re.IGNORECASE)
        return flow

    merged_df["Flow name"] = merged_df.apply(fix_ca_placeholder, axis=1)

    # --- 5️⃣ Unit conversions
    conversion_factors = {
        ("m3", "cubic meter"): 1.0,
        ("kg", "kilogram"): 1.0,
        ("tonnes", "kilogram"): 1000.0,
        ("t", "kilogram"): 1000.0,
        ("kilogram", "tonne"): 0.001,
        ("g", "kilogram"): 0.001,
        ("grams", "kilogram"): 0.001,
        ("mg", "kilogram"): 1e-6,
        ('tco2eq', 'kilogram'): 1000.0
    }

    def convert_value(row):
        val = row["value_normalized"]
        original_unit = str(row["unit"]).strip().lower()
        target_unit = str(row["Unit"]).strip().lower()
        key = (original_unit, target_unit)
        if pd.isna(val):
            return None
        elif key in conversion_factors:
            return val * conversion_factors[key]
        elif original_unit == target_unit:
            return val
        else:
            print(f"⚠️ No conversion defined for {original_unit} → {target_unit} (flow: {row['substance_name']})")
            return val  # keep unconverted if unknown

    merged_df["value_converted"] = merged_df.apply(convert_value, axis=1)

    # --- 6️⃣ Final formatting
    result_df = merged_df[[
        "activity_name",
        "functional_unit",
        "site_id",
        "substance_name",
        "province",
        "Flow name",
        "Compartments",
        "value_converted",
        "unit", # original unit
        "Unit", # target unit
        "DB_to_map"
    ]].rename(columns={
        "value_converted": "Amount",
        "Unit": "Unit",
        "Flow name": "Flow Name",
        "DB_to_map": "Database"
    })

    print(f"✅ Mapped {len(result_df)} biosphere flows ({len(result_df['Flow Name'].unique())} unique flows).")

    return result_df