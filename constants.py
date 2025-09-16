# Endpoints
IMPACT_METHODS_EP = {
'Total HH': ('IMPACT World+ Damage 2.1_regionalized for ecoinvent v3.10','Human health', 'Total human health'),
'Total EQ': ('IMPACT World+ Damage 2.1_regionalized for ecoinvent v3.10','Ecosystem quality', 'Total ecosystem quality'),
}


# Midpoints for ecosystem quality
IMPACT_METHODS_MP_EQ = {
    'Climate change EQ LT': ('IMPACT World+ Damage 2.1_regionalized for ecoinvent v3.10', 'Ecosystem quality', 'Climate change, ecosystem quality, long term'),
    'Climate change EQ ST': ('IMPACT World+ Damage 2.1_regionalized for ecoinvent v3.10', 'Ecosystem quality', 'Climate change, ecosystem quality, short term'),
    'Fisheries impact': ('IMPACT World+ Damage 2.1_regionalized for ecoinvent v3.10', 'Ecosystem quality', 'Fisheries impact'),
    'Freshwater acidification': ('IMPACT World+ Damage 2.1_regionalized for ecoinvent v3.10', 'Ecosystem quality', 'Freshwater acidification'),
    'Freshwater ecotoxicity LT': ('IMPACT World+ Damage 2.1_regionalized for ecoinvent v3.10', 'Ecosystem quality', 'Freshwater ecotoxicity, long term'),
    'Freshwater ecotoxicity ST': ('IMPACT World+ Damage 2.1_regionalized for ecoinvent v3.10', 'Ecosystem quality', 'Freshwater ecotoxicity, short term'),
    'Freshwater eutrophication': ('IMPACT World+ Damage 2.1_regionalized for ecoinvent v3.10', 'Ecosystem quality', 'Freshwater eutrophication'),
    'Ionizing radiations EQ': ('IMPACT World+ Damage 2.1_regionalized for ecoinvent v3.10', 'Ecosystem quality', 'Ionizing radiations, ecosystem quality'),
    'Land occupation biodiversity': ('IMPACT World+ Damage 2.1_regionalized for ecoinvent v3.10', 'Ecosystem quality', 'Land occupation, biodiversity'),
    'Land transformation biodiversity': ('IMPACT World+ Damage 2.1_regionalized for ecoinvent v3.10', 'Ecosystem quality', 'Land transformation, biodiversity'),
    'Marine acidification LT': ('IMPACT World+ Damage 2.1_regionalized for ecoinvent v3.10', 'Ecosystem quality', 'Marine acidification, long term'),
    'Marine acidification ST': ('IMPACT World+ Damage 2.1_regionalized for ecoinvent v3.10', 'Ecosystem quality', 'Marine acidification, short term'),
    'Marine ecotoxicity LT': ('IMPACT World+ Damage 2.1_regionalized for ecoinvent v3.10', 'Ecosystem quality', 'Marine ecotoxicity, long term'),
    'Marine ecotoxicity ST': ('IMPACT World+ Damage 2.1_regionalized for ecoinvent v3.10', 'Ecosystem quality', 'Marine ecotoxicity, short term'),
    'Marine eutrophication': ('IMPACT World+ Damage 2.1_regionalized for ecoinvent v3.10', 'Ecosystem quality', 'Marine eutrophication'),
    'Photochemical ozone EQ': ('IMPACT World+ Damage 2.1_regionalized for ecoinvent v3.10', 'Ecosystem quality', 'Photochemical ozone formation, ecosystem quality'),
    'Terrestrial acidification': ('IMPACT World+ Damage 2.1_regionalized for ecoinvent v3.10', 'Ecosystem quality', 'Terrestrial acidification'),
    'Terrestrial ecotoxicity LT': ('IMPACT World+ Damage 2.1_regionalized for ecoinvent v3.10', 'Ecosystem quality', 'Terrestrial ecotoxicity, long term'),
    'Terrestrial ecotoxicity ST': ('IMPACT World+ Damage 2.1_regionalized for ecoinvent v3.10', 'Ecosystem quality', 'Terrestrial ecotoxicity, short term'),
    'Thermally polluted water': ('IMPACT World+ Damage 2.1_regionalized for ecoinvent v3.10', 'Ecosystem quality', 'Thermally polluted water'),
    'Water availability freshwater': ('IMPACT World+ Damage 2.1_regionalized for ecoinvent v3.10', 'Ecosystem quality', 'Water availability, freshwater ecosystem'),
    'Water availability terrestrial': ('IMPACT World+ Damage 2.1_regionalized for ecoinvent v3.10', 'Ecosystem quality', 'Water availability, terrestrial ecosystem'),
}



## Aggregation for EQ
agg_mapping_eq = {'Freshwater ecotoxicity, long term': 'Freshwater ecotoxicity',
 'Freshwater ecotoxicity, short term': 'Freshwater ecotoxicity',
 'Terrestrial acidification': 'Terrestrial acidification',
 'Climate change, ecosystem quality, long term': 'Climate change',
 'Climate change, ecosystem quality, short term': 'Climate change',
 'Freshwater acidification': 'Freshwater acidification',
 'Terrestrial ecotoxicity, long term': 'Other ecotoxicities',
 'Marine ecotoxicity, long term': 'Other ecotoxicities',
 'Terrestrial ecotoxicity, short term': 'Other ecotoxicities',
 'Marine ecotoxicity, short term': 'Other ecotoxicities',
 'Land occupation, biodiversity': 'LULUC',
 'Land transformation, biodiversity': 'LULUC',
 'Water availability, freshwater ecosystem': 'Water',
 'Thermally polluted water': 'Water',
 'Water availability, terrestrial ecosystem': 'Water',
 'Marine eutrophication': 'Eutrophication',
 'Freshwater eutrophication': 'Eutrophication',
 'Marine acidification, long term': 'Marine acidification',
 'Marine acidification, short term': 'Marine acidification',
 'Photochemical ozone formation, ecosystem quality': 'Smog',
 'Fisheries impact': 'Fisheries impact',
 'Ionizing radiations, ecosystem quality': 'Ionizing radiations'}


# Midpoints for human health
IMPACT_METHODS_MP_HH = {
    'Climate change HH LT': ('IMPACT World+ Damage 2.1_regionalized for ecoinvent v3.10', 'Human health', 'Climate change, human health, long term'),
    'Climate change HH ST': ('IMPACT World+ Damage 2.1_regionalized for ecoinvent v3.10', 'Human health', 'Climate change, human health, short term'),
    'Human toxicity cancer LT': ('IMPACT World+ Damage 2.1_regionalized for ecoinvent v3.10', 'Human health', 'Human toxicity cancer, long term'),
    'Human toxicity cancer ST': ('IMPACT World+ Damage 2.1_regionalized for ecoinvent v3.10', 'Human health', 'Human toxicity cancer, short term'),
    'Human toxicity non-cancer LT': ('IMPACT World+ Damage 2.1_regionalized for ecoinvent v3.10', 'Human health', 'Human toxicity non-cancer, long term'),
    'Human toxicity non-cancer ST': ('IMPACT World+ Damage 2.1_regionalized for ecoinvent v3.10', 'Human health', 'Human toxicity non-cancer, short term'),
    'Ionizing radiations HH': ('IMPACT World+ Damage 2.1_regionalized for ecoinvent v3.10', 'Human health', 'Ionizing radiations, human health'),
    'Ozone layer depletion': ('IMPACT World+ Damage 2.1_regionalized for ecoinvent v3.10', 'Human health', 'Ozone layer depletion'),
    'Particulate matter formation': ('IMPACT World+ Damage 2.1_regionalized for ecoinvent v3.10', 'Human health', 'Particulate matter formation'),
    'Photochemical ozone HH': ('IMPACT World+ Damage 2.1_regionalized for ecoinvent v3.10', 'Human health', 'Photochemical ozone formation, human health'),
    'Water availability HH': ('IMPACT World+ Damage 2.1_regionalized for ecoinvent v3.10', 'Human health', 'Water availability, human health'),
}


## Aggregation for HH
agg_mapping_hh = {'Climate change, human health, long term': 'Climate change',
 'Climate change, human health, short term': 'Climate change',
 'Human toxicity cancer, long term': 'Human toxicity, cancer',
 'Human toxicity cancer, short term': 'Human toxicity, cancer',
 'Human toxicity non-cancer, long term': 'Human toxicity, non-cancer',
 'Human toxicity non-cancer, short term': 'Human toxicity, non-cancer',
 'Ionizing radiations, human health': 'Ionizing radiations',
 'Ozone layer depletion': 'Ozone layer depletion',
 'Particulate matter formation': 'Particulate matter',
 'Photochemical ozone formation, human health': 'Smog',
 'Water availability, human health': 'Water availability'}


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