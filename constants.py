# (mineral name: activity name, reference product, location)
IW_MD_DAMAGE = {

}

IW_ED_DAMAGE = {

}

CA_provinces = {
    'Alberta': 'CA-AB',
    'British Columbia': 'CA-BC',
    'Manitoba': 'CA-MB',
    'New Brunswick': 'CA-NB',
    'Newfoundland and Labrador': 'CA-NL',
    'Nova Scotia': 'CA-NS',
    'Ontario': 'CA-ON',
    'Prince Edward Island': 'CA-PE',
    'Quebec': 'CA-QC',
    'Saskatchewan': 'CA-SK',
    'Northwest Territories': 'CA-NT',
    'Nunavut': 'CA-NU',
    'Yukon': 'CA-YT'
}

# --- Direct energy units → MJ ---
UNIT_TO_MJ = {
    'mj':   1.0,
    'gj':   1_000.0,
    'tj':   1_000_000.0,
    'j':    1e-6,
    'wh':   0.0036,
    'kwh':  3.6,
    'mwh':  3_600.0,
    'gwh':  3_600_000.0,
}

# --- Volume unit multipliers (to liters) ---
VOLUME_TO_L = {
    'l': 1.0, 'liter': 1.0, 'litre': 1.0, 'liters': 1.0, 'litres': 1.0,
    'kl': 1_000.0, 'kiloliter': 1_000.0, 'kilolitre': 1_000.0,
    'ml': 1_000_000.0, 'megaliter': 1_000_000.0, 'megalitre': 1_000_000.0,
}

CUBIC_M_TO_M3 = {'m3': 1.0, 'm^3': 1.0, 'cubicmeter': 1.0, 'cubicmeters': 1.0}

# --- Default LHVs (edit with site/company data whenever you can) ---
DEFAULT_LHV = {
    'diesel':      {'MJ/kg': 43.0, 'MJ/L': 38.6, 'density_kg_per_L': 0.835},
    'gasoline':    {'MJ/kg': 44.0, 'MJ/L': 34.2, 'density_kg_per_L': 0.745},
    'heavy_fuel_oil': {'MJ/kg': 40.5, 'density_kg_per_L': 0.98},
    'coal':        {'MJ/kg': 25.0},
    # Per your note: assume LNG when volume units used for “natural gas”
    'natural_gas': {'MJ/m3': 38.0, 'MJ/L': 22.5, 'density_kg_per_L': 0.7},
    'propane':     {'MJ/kg': 46.4, 'MJ/L': 25.3, 'density_kg_per_L': 0.493},
    'electricity': {'MJ/kWh': 3.6},  # direct units (kWh/MWh/GWh) handled by UNIT_TO_MJ
    'explosives': {'MJ/kg': 4.0},  # TNT equivalents
}

# Default densities (kg/L)
DEFAULT_DENSITY = {
    # Oils & lubricants family
    'lubricants': 0.88,
    'hydraulic oil': 0.88,
    'transmission oil': 0.88,
    'motor oil': 0.88,
    'drill oil': 0.88,
    'compressor oil': 0.88,

    # Acids (typical commercial concentrations)
    'sulfuric acid (h2so4)': 1.84,    # ~98%
    'hydrochloric acid (hcl)': 1.19,  # ~37%
    'nitric acid (hno3)': 1.51,       # ~68–70%

}

# --- Subflow canonicalization (aliases + strip pipe suffixes) ---
SUBFLOW_ALIASES = {
    'petrol': 'gasoline',
    'heavy fuel oil': 'heavy_fuel_oil',
    'hfo': 'heavy_fuel_oil',
    'natural gas': 'natural_gas',
    'explosive': 'explosives',
    'explosives': 'explosives'
}

# Canonicalize names (left part before '|', lowercased)
ALIASES = {
    'petrol': 'gasoline',
    'grindingmedia': 'grinding media',
    '3/4\'\'balls': 'grinding media',
    '2\'\'balls': 'grinding media',
    '2.5\'\'balls': 'grinding media',
    '5.5\'\'balls': 'grinding media',
    'polyfrothh57': 'polyfroth h57',
    'antiscalant': 'anti-scalant',
}