# ======================================================
# Energy equivalence = How much *input fuel* energy is required to produce 1 MJ of delivered energy
# ======================================================
ENERGY_EQUIVALENCE = {
    ('electricity', 'diesel'): {'efficiency': 0.35, 'MJ/kg': 43.0}, #(diesel generator efficiency ~35%)
    ('electricity', 'natural_gas'): {'efficiency': 0.40, 'MJ/m3': 35.2},
    ('electricity', 'propane'): {'efficiency': 0.38, 'MJ/kg': 46.4},
}

# ======================================================
# Energy --> Physical constants
# ======================================================
UNIT_TO_MJ = {
    'mj':   1.0, 'gj':   1_000.0, 'tj':   1_000_000.0, 'j': 1e-6,
    'wh':   0.0036, 'kwh':  3.6, 'mwh':  3_600.0, 'gwh':  3_600_000.0,
}

# --- Volume unit multipliers (to liters) ---
VOLUME_TO_L = {
    'l': 1.0, 'liter': 1.0, 'litre': 1.0, 'liters': 1.0, 'litres': 1.0,
    'kl': 1_000.0, 'kiloliter': 1_000.0, 'kilolitre': 1_000.0,
    'ml': 1_000_000.0, 'megaliter': 1_000_000.0, 'megalitre': 1_000_000.0,
    'gallon': 3.78541, 'gallons': 3.78541,
}

CUBIC_M_TO_M3 = {'m3': 1.0, 'm^3': 1.0, 'cubicmeter': 1.0, 'cubicmeters': 1.0, 'kl': 1.0} # kl = cubic meter for gases as assumption

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

# ======================================================
# Energy content (LHV) and densities
# ======================================================
DEFAULT_LHV = {
    'diesel':       {'MJ/kg': 43.0, 'MJ/L': 38.6, 'density_kg_per_L': 0.835},
    'gasoline':     {'MJ/kg': 44.0, 'MJ/L': 34.2, 'density_kg_per_L': 0.745},
    'heavy_fuel_oil': {'MJ/kg': 40.5, 'MJ/L': 39.69, 'density_kg_per_L': 0.98},
    'coal':         {'MJ/kg': 25.0},
    'natural_gas':  {'MJ/m3': 38.0, 'MJ/L': 22.5, 'density_kg_per_L': 0.7},
    'propane':      {'MJ/kg': 46.4, 'MJ/L': 25.3, 'density_kg_per_L': 0.493},
    'electricity':  {'MJ/kWh': 3.6},
    'explosives':   {'MJ/kg': 4.0},
    'coke':         {'MJ/kg': 28.0},
    'wood':         {'MJ/kg': 16.0},
    'acetylene':    {'MJ/kg': 48.0},
    'used_oil':     {'MJ/kg': 42.0},
    'biodiesel':    {'MJ/kg': 37.4, 'density_kg_per_L': 0.877},
    'naphtha':      {'MJ/kg': 44.9, 'density_kg_per_L': 0.725},
    "kerosene":     {'MJ/kg': 43.1},
    "aviation fuel": {'MJ/kg': 43.1},
    "anfo":         {'MJ/kg': 2.3},
    "emulsion":     {'MJ/kg': 3.7},
    "explosive_default": {'MJ/kg': 3.0},
}

# Densities (kg per m3)
density_kg_per_m3 = {
    "biodiesel": 877.0,  # ~0.877 kg/L => 877 kg/m3 (Sandia/AFDC)
    "naphtha": 725.0,  # ~0.725 kg/L => 725 kg/m3 (kerone)
}

# --- Default densities (kg/L) for common liquids where LHV is not defined ---
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

# ======================================================
# Canonical subflow names and aliases
# ======================================================
SUBFLOW_ALIASES = {
    'petrol': 'gasoline',
    'heavy fuel oil': 'heavy_fuel_oil',
    'hfo': 'heavy_fuel_oil',
    'natural gas': 'natural_gas',
    'explosive': 'explosives',
    'lpg': 'propane',
    'surface/underground_emulsion_&_anfo': 'explosives',
    'grinding_media': 'explosives',
    'total_blasting_agents_used_e.g._anfo': 'explosives',
    'grindingmedia': 'grinding media',
    '3/4\'\'balls': 'grinding media',
    '2\'\'balls': 'grinding media',
    '2.5\'\'balls': 'grinding media',
    '5.5\'\'balls': 'grinding media',
    'polyfrothh57': 'polyfroth h57',
    'antiscalant': 'anti-scalant',
}

nrj_subflow = {
    # electricity
    'Electricity consumption|Generated on-site': 'Electricity',
    'Electricity consumption': 'Electricity',
    'Electricity consumption|Grid electricity': 'Electricity',
    'Electricity consumption|Not specified': 'Electricity',
    'Electricity consumption|Non-renewable electricity use': 'Electricity',
    'Solar': 'Electricity',

    # Fuels
    'Diesel': 'Diesel',
    'Diesel|Mobile equipment': 'Diesel',
    'Diesel|Stationary equipment': 'Diesel',
    'Gasoline': 'Gasoline',
    'Gasoline|Mobile equipment': 'Gasoline',
    'Petrol': 'Gasoline',
    'Oil': 'Gasoline',  # usually refers to gasoline
    'Light Fuel & Gasoline': 'Gasoline',
    'Lubricating Oils & Greases': 'Lubricants',
    'Biodiesel': 'Diesel',
    'Propane': 'LPG-Propane',
    'LPG': 'LPG-Propane',
    'LPG|Mobile equipment': 'LPG-Propane',
    'LPG|Stationary equipment': 'LPG-Propane',
    'Acetylene': 'LPG-Propane',
    'Natural gas': 'Natural gas',
    'Naphta': 'Naphtha',  # spelling
    'Aviation fuel': 'Aviation fuel',
    'Non-renewable fuel use': 'Diesel',

    # Explosives
    'Explosives': 'Explosives',
    'Total blasting agents used e.g. anfo': 'Explosives',
    'ANFO': 'Explosives',
    'Emulsion ANFO': 'Explosives',
    'Emulsions': 'Explosives',
    'Emulsion': 'Explosives',
    'Dynamite': 'Explosives',
    'Ammonium nitrate': 'Explosives',  # (treat as energy only if you ANFO-equivalize)

    # Others
    'Used oil': 'Other',  # usually MATERIAL (lubricants); map to energy only if burned
    'Other': 'Other',
}

# ======================================================
# LCA stuff
# ======================================================
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

agg_mapping_eq = {
 'Freshwater ecotoxicity LT': 'Freshwater ecotoxicity',
 'Freshwater ecotoxicity ST': 'Freshwater ecotoxicity',
 'Terrestrial acidification': 'Terrestrial acidification',
 'Climate change EQ LT': 'Climate change',
 'Climate change EQ ST': 'Climate change',
 'Freshwater acidification': 'Freshwater acidification',
 'Terrestrial ecotoxicity LT': 'Terrestrial ecotoxicity',
 'Marine ecotoxicity LT': 'Marine ecotoxicity',
 'Terrestrial ecotoxicity ST': 'Terrestrial ecotoxicity',
 'Marine ecotoxicity ST': 'Marine ecotoxicity',
 'Land occupation biodiversity': 'Land occupation',
 'Land transformation biodiversity': 'Land transformation',
 'Water availability freshwater ecosystem': 'Freshwater availability',
 'Thermally polluted water': 'Thermally polluted water',
 'Water availability terrestrial ecosystem': 'Terrestrial water availability',
 'Marine eutrophication': 'Eutrophication',
 'Freshwater eutrophication': 'Eutrophication',
 'Marine acidification LT': 'Marine acidification',
 'Marine acidification ST': 'Marine acidification',
 'Photochemical ozone EQ': 'Photochemical ozone formation',
 'Fisheries impact': 'Fisheries impact',
 'Ionizing radiations EQ': 'Ionizing radiations'
}

agg_mapping_hh = {
 'Climate change HH LT': 'Climate change',
 'Climate change HH ST': 'Climate change',
 'Human toxicity cancer LT': 'Human toxicity, cancer',
 'Human toxicity cancer ST': 'Human toxicity, cancer',
 'Human toxicity non-cancer LT': 'Human toxicity, non-cancer',
 'Human toxicity non-cancer ST': 'Human toxicity, non-cancer',
 'Ionizing radiations HH': 'Ionizing radiations',
 'Ozone layer depletion': 'Ozone layer depletion',
 'Particulate matter formation': 'Particulate matter',
 'Photochemical ozone HH': 'Photochemical ozone formation',
 'Water availability HH': 'Water availability'
}

# ======================================================
# CA provinces
# ======================================================
CA_provinces = {
    'Alberta': 'CA-AB',
    'British Columbia': 'CA-BC',
    'Manitoba': 'CA-MB',
    'New Brunswick': 'CA-NB',
    'Newfoundland and Labrador': 'CA-NF',
    'Nova Scotia': 'CA-NS',
    'Ontario': 'CA-ON',
    'Prince Edward Island': 'CA-PE',
    'Quebec': 'CA-QC',
    'Saskatchewan': 'CA-SK',
    'Northwest Territories': 'CA-NT',
    'Nunavut': 'CA-NU',
    'Yukon': 'CA-YK'
}

# ======================================================
# Metal stuff
# ======================================================
metal_map = {
    # Base/industrial metals
    "Aluminium": "Al",
    "Copper": "Cu",
    "Iron": "Fe",
    "Lead": "Pb",
    "Nickel": "Ni",
    "Tin": "Sn",
    "Zinc": "Zn",
    "Chromium": "Cr",
    "Manganese": "Mn",
    "Titanium": "Ti",
    "Cobalt": "Co",
    "Magnesium": "Mg",
    "Vanadium": "V",
    "Molybdenum": "Mo",
    "Tungsten": "W",
    "Mercury": "Hg",
    "Cadmium": "Cd",
    "Antimony": "Sb",
    "Bismuth": "Bi",
    "Arsenic": "As",
    "Cesium": "Cs",
    "Uranium": "U",

    # Precious metals
    "Gold": "Au",
    "Silver": "Ag",
    "Platinum": "Pt",
    "Palladium": "Pd",
    "Rhodium": "Rh",
    "Ruthenium": "Ru",
    "Iridium": "Ir",
    "Osmium": "Os",

    # Rare earth elements (REEs)
    "Lanthanum": "La",
    "Cerium": "Ce",
    "Praseodymium": "Pr",
    "Neodymium": "Nd",
    "Promethium": "Pm",
    "Samarium": "Sm",
    "Europium": "Eu",
    "Gadolinium": "Gd",
    "Terbium": "Tb",
    "Dysprosium": "Dy",
    "Holmium": "Ho",
    "Erbium": "Er",
    "Thulium": "Tm",
    "Ytterbium": "Yb",
    "Lutetium": "Lu",
    "Yttrium": "Y",

    # Other minor metals
    "Beryllium": "Be",
    "Gallium": "Ga",
    "Indium": "In",
    "Tantalum": "Ta",
    "Niobium": "Nb",
    "Rhenium": "Re",
    "Scandium": "Sc",
    "Zirconium": "Zr",
    "Hafnium": "Hf",
    "Tellurium": "Te",
    "Selenium": "Se",
    "Lithium": "Li",
    "Strontium": "Sr",
    "Boron": "B",

    # Custom
    "Platinum group metals": "PGMs",
}