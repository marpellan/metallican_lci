# ======================================================
# LCA stuff
# ======================================================
# Total endpoints
IMPACT_METHODS_EP = {
'Total HH': ('IMPACT World+ Damage 2.1_regionalized for ecoinvent v3.10','Human health', 'Total human health'),
'Total EQ': ('IMPACT World+ Damage 2.1_regionalized for ecoinvent v3.10','Ecosystem quality', 'Total ecosystem quality'),
}

# Damages for ecosystem quality
IMPACT_METHODS_EQ_damages = {
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

# Damages for human health
IMPACT_METHODS_HH_damages = {
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

# Selected MPs
IMPACT_METHODS_MIDPOINT = { # 10 indicators?
#'Climate change LT': ('IMPACT World+ Midpoint 2.1_regionalized for ecoinvent v3.10','Midpoint','Climate change, long term'),
'Climate change ST': ('IMPACT World+ Midpoint 2.1_regionalized for ecoinvent v3.10','Midpoint','Climate change, short term'),
#'Fossil and nuclear energy use': ('IMPACT World+ Midpoint 2.1_regionalized for ecoinvent v3.10','Midpoint','Fossil and nuclear energy use'),
#'Freshwater acidification': ('IMPACT World+ Midpoint 2.1_regionalized for ecoinvent v3.10','Midpoint','Freshwater acidification'),
#'Terrestrial acidification': ('IMPACT World+ Midpoint 2.1_regionalized for ecoinvent v3.10','Midpoint','Terrestrial acidification'),
#'Freshwater ecotoxicity': ('IMPACT World+ Midpoint 2.1_regionalized for ecoinvent v3.10','Midpoint','Freshwater ecotoxicity'),
#'Freshwater eutrophication': ('IMPACT World+ Midpoint 2.1_regionalized for ecoinvent v3.10','Midpoint','Freshwater eutrophication'),
#'Human toxicity cancer': ('IMPACT World+ Midpoint 2.1_regionalized for ecoinvent v3.10','Midpoint','Human toxicity cancer'),
#'Human toxicity non-cancer': ('IMPACT World+ Midpoint 2.1_regionalized for ecoinvent v3.10','Midpoint','Human toxicity non-cancer' ),
#'Ionizing radiations': ('IMPACT World+ Midpoint 2.1_regionalized for ecoinvent v3.10','Midpoint','Ionizing radiations'),
'Land occupation biodiversity': ('IMPACT World+ Midpoint 2.1_regionalized for ecoinvent v3.10','Midpoint','Land occupation, biodiversity'),
'Land transformation biodiversity': ('IMPACT World+ Midpoint 2.1_regionalized for ecoinvent v3.10','Midpoint','Land transformation, biodiversity'),
#'Marine eutrophication': ('IMPACT World+ Midpoint 2.1_regionalized for ecoinvent v3.10','Midpoint','Marine eutrophication' ),
#'Mineral resources use': ('IMPACT World+ Midpoint 2.1_regionalized for ecoinvent v3.10','Midpoint','Mineral resources use'),
#'Ozone layer depletion': ('IMPACT World+ Midpoint 2.1_regionalized for ecoinvent v3.10','Midpoint','Ozone layer depletion'),
'Particulate matter formation': ('IMPACT World+ Midpoint 2.1_regionalized for ecoinvent v3.10','Midpoint','Particulate matter formation'),
#'Photochemical ozone formation': ('IMPACT World+ Midpoint 2.1_regionalized for ecoinvent v3.10','Midpoint','Photochemical ozone formation'),
'Water scarcity': ('IMPACT World+ Midpoint 2.1_regionalized for ecoinvent v3.10','Midpoint','Water scarcity'),
}


agg_mapping_eq = {
 'Freshwater ecotoxicity LT': 'Freshwater ecotoxicity',
 'Freshwater ecotoxicity ST': 'Freshwater ecotoxicity',
 'Terrestrial acidification': 'Terrestrial acidification',
 'Climate change EQ LT': 'Climate change',
 'Climate change EQ ST': 'Climate change',
 'Freshwater acidification': 'Freshwater acidification',
 'Terrestrial ecotoxicity LT': 'Other',
 'Marine ecotoxicity LT': 'Other',
 'Terrestrial ecotoxicity ST': 'Other',
 'Marine ecotoxicity ST': 'Other',
 'Land occupation biodiversity': 'Land occupation',
 'Land transformation biodiversity': 'Land transformation',
 'Water availability freshwater ecosystem': 'Other',
 'Thermally polluted water': 'Other',
 'Water availability terrestrial ecosystem': 'Other',
 'Marine eutrophication': 'Other',
 'Freshwater eutrophication': 'Other',
 'Marine acidification LT': 'Marine acidification',
 'Marine acidification ST': 'Marine acidification',
 'Photochemical ozone EQ': 'Other',
 'Fisheries impact': 'Other',
 'Ionizing radiations EQ': 'Other'
}

agg_mapping_hh = {
 'Climate change HH LT': 'Climate change',
 'Climate change HH ST': 'Climate change',
 'Human toxicity cancer LT': 'Human toxicity, cancer',
 'Human toxicity cancer ST': 'Human toxicity, cancer',
 'Human toxicity non-cancer LT': 'Other',
 'Human toxicity non-cancer ST': 'Other',
 'Ionizing radiations HH': 'Other',
 'Ozone layer depletion': 'Other',
 'Particulate matter formation': 'Particulate matter',
 'Photochemical ozone HH': 'Other',
 'Water availability HH': 'Water availability'
}


agg_st_lt_eq = {
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
 'Water availability freshwater ecosystem': 'Water availability',
 'Thermally polluted water': 'Thermally polluted water',
 'Water availability terrestrial ecosystem': 'Water availability',
 'Marine eutrophication': 'Marine eutrophication',
 'Freshwater eutrophication': 'Freshwater eutrophication',
 'Marine acidification LT': 'Marine acidification',
 'Marine acidification ST': 'Marine acidification',
 'Photochemical ozone EQ': 'Photochemical ozone',
 'Fisheries impact': 'Fisheries impact',
 'Ionizing radiations EQ': 'Ionizing radiations'
}


agg_st_lt_hh = {
 'Climate change HH LT': 'Climate change',
 'Climate change HH ST': 'Climate change',
 'Human toxicity cancer LT': 'Human toxicity, cancer',
 'Human toxicity cancer ST': 'Human toxicity, cancer',
 'Human toxicity non-cancer LT': 'Human toxicity non-cancer',
 'Human toxicity non-cancer ST': 'Human toxicity non-cancer',
 'Ionizing radiations HH': 'Ionizing radiations',
 'Ozone layer depletion': 'Ozone layer depletion',
 'Particulate matter formation': 'Particulate matter formation',
 'Photochemical ozone HH': 'Photochemical ozone',
 'Water availability HH': 'Water availability'
}