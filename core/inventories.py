# ======================================================
# Ecoinvent
# ======================================================
INVENTORIES_CA_ei = {
    "Cu concentrate from Au mine": ("gold-silver mine operation and beneficiation", "copper concentrate, sulfide ore", 'CA-QC'),
    "Cu concentrate from Cu mine": ("copper mine operation and beneficiation, sulfide ore", "copper concentrate, sulfide ore", 'CA'),
    "Au-Ag ingot": ("gold-silver mine operation and beneficiation", "gold-silver, ingot", "CA-QC"),
    "Pb concentrate": ("gold-silver mine operation and beneficiation", "lead concentrate", "CA-QC"),
    "Zn concentrate": ("gold-silver mine operation and beneficiation", "zinc concentrate", "CA-QC"),
    "Fe concentrate": ("iron ore mine operation and beneficiation", "iron ore concentrate", "CA-QC"),
    "Ni concentrate": ("nickel mine operation and benefication to nickel concentrate, 16% Ni", "nickel concentrate, 16% Ni", "CA-QC"), #typo in EI
    "U yellowcake": ("uranium production, in yellowcake", "uranium, in yellowcake", "RoW")
}


# ======================================================
# Regioinvent
# ======================================================
INVENTORIES_CA_reg = {
    "Cu concentrate from Au mine": ("gold-silver mine operation and beneficiation", "copper concentrate, sulfide ore", 'CA'),
    "Cu concentrate from Cu mine": ("copper mine operation and beneficiation, sulfide ore", "copper concentrate, sulfide ore", 'CA'),
    "Au-Ag ingot": ("gold-silver mine operation and beneficiation", "gold-silver, ingot", "CA"),
    "Pb concentrate": ("gold-silver mine operation and beneficiation", "lead concentrate", "CA"),
    "Zn concentrate": ("gold-silver mine operation and beneficiation", "zinc concentrate", "CA"),
    "Fe concentrate": ("iron ore mine operation and beneficiation", "iron ore concentrate", "CA"),
    "Ni concentrate": ("nickel mine operation and benefication to nickel concentrate, 16% Ni", "nickel concentrate, 16% Ni", "CA"), #typo in EI
    "U yellowcake": ("uranium production, in yellowcake", "uranium, in yellowcake", "CA")

}


# ======================================================
# Lai et al 2025
# ======================================================
INVENTORIES_Lai_Cu = { # From DB - 'Copper'
    #"Cu concentrate (Australia)": ("[M+C] Copper (Cu) ore mining and concentration (Australia)", "Cu concentrate", 'AU'), # almost no data
    #"Cu concentrate (US)": ("[M+C] Copper (Cu) ore mining and concentration (United States)", "Cu concentrate", 'US'), # almost no data
    "Cu concentrate (Sweden)": ("[M+C] Copper (Cu) ore mining and concentration (Sweden)", "Cu concentrate", "SE"),
    "Cu concentrate (China)": ("[M+C] Copper (Cu) ore mining and concentration (China)", "Cu concentrate", "CB"),
    "Cu concentrate (Chile)": ("[M+C] Copper (Cu) ore mining and concentration (Chile)", "Cu concentrate", "CL")
}


INVENTORIES_Lai_Fe = { # FROM DB = 'Iron'
    "Fe concentrate (Canada)": ("[M+C] Iron (Fe) ore mining and concentration (Canada)", "Fe concentrate", 'CA'), # water and PM10 in biosphere data
    "Fe concentrate (Brazil)": ("[M+C] Iron (Fe) ore mining and concentration (Brazil)", "Fe concentrate", 'BR'),
    "Fe concentrate (Ukraine)": ("[M+C] Iron (Fe) ore mining and concentration (Ukraine)", "Fe concentrate", 'UA'), # almost no biosphere data
    "Fe concentrate (Australia)": ("[M+C] Iron (Fe) ore mining and concentration (Australia)", "Fe concentrate", 'AU'), # only water in biosphere data
    "Fe concentrate (South Africa)": ("[M+C] Iron (Fe) ore mining and concentration (South Africa)", "Fe concentrate", 'ZA'), # water and PM10 in biosphere data
    "Fe concentrate (China)": ("[M+C] Iron (Fe) ore mining and concentration (China)", "Fe concentrate", 'CN'),
    "Fe concentrate (Global)": ("Iron ore mining and concentration - Global market", "Fe concentrate", 'GLO') # No biosphere data
}


#INVENTORIES_Lai_Ni = { # FROM DB = 'Nickel' a lot of them are not included because almost empty
#    "Ni concentrate (CN)": ("[M+C] Nickel (Ni) ore mining and concentration (GFEM route)", "Ni concentrate", 'CN'),

#}

# ======================================================
# MetalliCan
# ======================================================
INVENTORIES_metallican = {
# Cu
"Copper Mountain": ("Open-pit mining and beneficiation at Copper Mountain", "Cu concentrate", "CA-BC"),
"Gibraltar (Cu)": ("Open-pit mining and beneficiation at Gibraltar", "Cu concentrate", "CA-BC"),
"Highland Valley (Cu)": ("Open-pit mining and beneficiation at Highland Valley", "Cu concentrate", "CA-BC"),
"Mount Milligan": ("Open-pit mining and beneficiation at Mount Milligan", "Cu concentrate", "CA-BC"),
"Mount Polley": ("Open-pit mining and beneficiation at Mount Polley", "Cu concentrate", "CA-BC"),
#"Red Chris": ("Open-pit mining and beneficiation at Red Chris", "Cu concentrate", "CA-BC"),
"Goldex (Cu)": ("Underground mining and beneficiation at Goldex", "Cu concentrate", "CA-QC"),
"Kidd Creek": ("Underground mining and beneficiation at Kidd Creek", "Cu concentrate", "CA-ON"),
"LaRonde (Cu)": ("Underground mining and beneficiation at LaRonde", "Cu concentrate", "CA-QC"),
"New Afton": ("Underground mining and beneficiation at New Afton", "Cu concentrate", "CA-BC"),
"Snow Lake (Cu)": ("Underground mining and beneficiation at Snow Lake", "Cu concentrate", "CA-MB"),


# Dore
"Casa Berardi": ("Open-pit and underground mining and beneficiation at Casa Berardi", "Doré", "CA-QC"),
#"Fox Complex": ("Open-pit and underground mining and beneficiation at Fox Complex", "Doré", "CA-ON"), # No GHGs data
"Hemlo": ("Open-pit and underground mining and beneficiation at Hemlo (Williams)", "Doré", "CA-ON"),
"Meliadine": ("Open-pit and underground mining and beneficiation at Meliadine", "Doré", "CA-NU"),
"Rainy River": ("Open-pit and underground mining and beneficiation at Rainy River", "Doré", "CA-ON"),
"Meadowbank complex": ("Open-pit and underground mining at Meadowbank complex", "Doré", "CA-NU"),
"Porcupine complex": ("Open-pit and underground mining at Porcupine complex", "Doré", "CA-ON"),
"Canadian Malartic": ("Open-pit mining and beneficiation at Canadian Malartic", "Doré", "CA-QC"),
"Detour Lake": ("Open-pit mining and beneficiation at Detour Lake", "Doré", "CA-ON"),
"Kiena": ("Open-pit mining and beneficiation at Kiena", "Doré", "CA-QC"), # No NPRI data but still some extracted
#"Magino": ("Open-pit mining and beneficiation at Magino", "Doré", "CA-ON"), No NPRI
#"Elk": ("Open-pit mining at Elk", "Doré", "CA-BC"),
"Brucejack": ("Underground mining and beneficiation at Brucejack", "Doré", "CA-BC"),
"Eagle River": ("Underground mining and beneficiation at Eagle River", "Doré", "CA-ON"),
"Éléonore": ("Underground mining and beneficiation at Éléonore", "Doré", "CA-QC"),
"Goldex (Au)": ("Underground mining and beneficiation at Goldex", "Doré", "CA-QC"),
"Island": ("Underground mining and beneficiation at Island", "Doré", "CA-ON"),
"LaRonde (Doré)": ("Underground mining and beneficiation at LaRonde", "Doré", "CA-QC"),
#"Lamaque": ("Underground mining and beneficiation at Lamaque", "Doré", "CA-QC"), # No NPRI
"Macassa": ("Underground mining and beneficiation at Macassa", "Doré", "CA-ON"),
"Musselwhite": ("Underground mining and beneficiation at Musselwhite", "Doré", "CA-ON"),
"Red Lake": ("Underground mining and beneficiation at Red Lake", "Doré", "CA-ON"),
"Seabee Gold Operation": ("Underground mining and beneficiation at Seabee Gold Operation", "Doré", "CA-SK"),
"Snow Lake (Doré)": ("Underground mining and beneficiation at Snow Lake", "Doré", "CA-MB"),
"Timmins Operation": ("Underground mining and beneficiation at Timmins Operation", "Doré", "CA-ON"),
"Westwood-Doyon": ("Underground mining and beneficiation at Westwood-Doyon", "Doré", "CA-QC"),

# Ni
"Nunavik Nickel": ("Open-pit and underground mining and beneficiation at Nunavik Nickel", "Ni concentrate", "CA-QC"),
"Thompson": ("Open-pit and underground mining and beneficiation at Thompson (T-1 and T-3)", "Ni concentrate", "CA-MB"),
"Voisey's Bay": ("Open-pit and underground mining and beneficiation at Voisey’s Bay", "Ni concentrate", "CA-NF"),
"Raglan": ("Underground mining and beneficiation at Raglan", "Ni concentrate", "CA-QC"),
#"Strathcona": ("and beneficiation at Strathcona", "Ni concentrate", "CA-ON"),
## Some others but we don't include them for now

# Mo
"Gibraltar (Mo)": ("Open-pit mining and beneficiation at Gibraltar", "Mo concentrate", "CA-BC"),
"Highland Valley (Mo)": ("Open-pit mining and beneficiation at Highland Valley", "Mo concentrate", "CA-BC"),

# Zn
"Kidd Creek (Zn)": ("Underground mining and beneficiation at Kidd Creek", "Zn concentrate", "CA-ON"),
"LaRonde (Zn)": ("Underground mining and beneficiation at LaRonde", "Zn concentrate", "CA-QC"),
"Snow Lake (Zn)": ("Underground mining and beneficiation at Snow Lake", "Zn concentrate", "CA-MB"),

# U
"Key Lake + McArthur River": ("Underground mining and beneficiation at Key Lake + McArthur River", "U concentrate", "CA-SK"),
"Cigar Lake + McClean Lake": ("Underground mining and beneficiation at Cigar Lake + McClean Lake", "U concentrate", "CA-SK"),

# Markets
"Market for Doré": ("Market for Doré", "Doré", "CA"),
"Market for Cu concentrate": ("Market for Cu concentrate", "Cu concentrate", "CA"),
"Market for Ni concentrate": ("Market for Ni concentrate", "Ni concentrate", "CA"),
"Market for U concentrate": ("Market for U concentrate", "U concentrate", "CA"),
"Market for Mo concentrate": ("Market for Mo concentrate", "Mo concentrate", "CA"),
"Market for Zn concentrate": ("Market for Zn concentrate", "Zn concentrate", "CA")

}

# INVENTORIES_market_metallican = {
# "Market for Doré": ("Market for Doré", "Doré", "CA"),
# "Market for Cu concentrate": ("Market for Cu concentrate", "Cu concentrate", "CA"),
# "Market for Ni concentrate": ("Market for Ni concentrate", "Ni concentrate", "CA"),
# "Market for U concentrate": ("Market for U concentrate", "U concentrate", "CA"),
# "Market for Mo concentrate": ("Market for Mo concentrate", "Mo concentrate", "CA"),
# "Market for Zn concentrate": ("Market for Zn concentrate", "Zn concentrate", "CA"),
# }