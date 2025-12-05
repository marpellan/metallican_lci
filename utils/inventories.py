# ======================================================
# Ecoinvent
# ======================================================
INVENTORIES_CA_ei = {
    "Cu concentrate from Au mine": ("gold-silver mine operation and beneficiation", "copper concentrate, sulfide ore", 'CA-QC'),
    "Cu concentrate from Cu mine": ("copper mine operation and beneficiation, sulfide ore", "copper concentrate, sulfide ore", 'CA'),
    "Au-Ag ingot": ("gold-silver mine operation and beneficiation", "gold-silver, ingot", "CA-QC"),
    "Pb concentrate": ("gold-silver mine operation and beneficiation", "lead concentrate", "CA-QC"),
    "Zn concentrate": ("gold-silver mine operation and beneficiation", "zinc concentrate", "CA-QC"),
    "Au refined": ("gold-silver mine operation with refinery", "gold", "CA-QC"),
    "Ag refined": ("gold-silver mine operation with refinery", "silver", "CA-QC"),
    "Fe concentrate": ("iron ore mine operation and beneficiation", "iron ore concentrate", "CA-QC"),
    "Ni concentrate": ("nickel mine operation and benefication to nickel concentrate, 16% Ni", "nickel concentrate, 16% Ni", "CA-QC"), #typo in EI
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
    "Au refined": ("gold-silver mine operation with refinery", "gold", "CA"),
    "Ag refined": ("gold-silver mine operation with refinery", "silver", "CA"),
    "Fe concentrate": ("iron ore mine operation and beneficiation", "iron ore concentrate", "CA"),
    "Ni concentrate": ("nickel mine operation and benefication to nickel concentrate, 16% Ni", "nickel concentrate, 16% Ni", "CA"), #typo in EI
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
INVENTORIES_metallican_ore = {
    "Brucejack": ("Au, Underground mining and beneficiation at Brucejack", "Gold, silver", "CA-BC"),
    "Canadian Malartic": ("Au and Ag, Open-pit mining and beneficiation at Canadian Malartic", "Gold, silver", "CA-QC"),
    "Casa Berardi": ("Au and Ag, Open-pit and underground mining and beneficiation at Casa Berardi", "Gold, silver", "CA-QC"),
    "Detour Lake": ("Au and Ag, Open-pit mining and beneficiation at Detour Lake", "Gold", "CA-ON"),
    "Eagle River": ("Au, Underground mining and beneficiation at Eagle River", "Gold", "CA-ON"),
    "Éléonore": ("Au, Underground mining and beneficiation at Éléonore", "Gold", "CA-QC"),
    "Goldex": ("Au and Ag and Zn, Underground mining and beneficiation at Goldex", "Gold, silver", "CA-QC"),
    "Hemlo (Williams)": ("Au, Open-pit and underground mining and beneficiation at Hemlo (Williams)", "Gold", "CA-ON"),
    "Kiena": ("Au, Open-pit mining and beneficiation at Kiena", "Gold", "CA-QC"),
    "Lamaque": ("Au, Underground mining and beneficiation at Lamaque", "Gold", "CA-QC"),
    "LaRonde": ("Au and Ag and Cu and Zn, Underground mining and beneficiation at LaRonde", "Gold, zinc, copper, silver, cadmium", "CA-QC"),
    "Macassa": ("Au, Underground mining and beneficiation at Macassa", "Gold, silver", "CA-ON"),
    "Meliadine": ("Au and Ag, Open-pit and underground mining and beneficiation at Meliadine", "Gold", "CA-NU"),
    "Musselwhite": ("Au, Underground mining and beneficiation at Musselwhite", "Gold, silver", "CA-ON"),
    "New Afton": ("Au and Cu, Underground mining and beneficiation at New Afton", "Gold, copper, silver", "CA-BC"),
    "Rainy River": ("Au, Open-pit and underground mining and beneficiation at Rainy River", "Gold, silver", "CA-ON"),
    "Red Chris": ("Au and Cu, Open-pit mining and beneficiation at Red Chris", "Gold, copper, silver", "CA-BC"),
    "Red Lake": ("Au, Underground mining and beneficiation at Red Lake", "Gold, silver", "CA-ON"),
    "Westwood-Doyon": ("Au, Underground mining and beneficiation at Westwood-Doyon", "Gold, silver", "CA-QC"),
    "Young-Davidson": ("Au, Underground mining and beneficiation at Young-Davidson", "Gold", "CA-ON"),
    "Meadowbank complex": ("Au and Ag, Open-pit and underground mining at Meadowbank complex", "Gold", "CA-NU"),
    "Porcupine complex": ("Au, Open-pit and underground mining at Porcupine complex", "Gold", "CA-ON"),
    "Timmins Operation": ("Au and Ag, Underground mining and beneficiation at Timmins Operation", "Gold", "CA-ON"),
    "Seabee Gold Operation": ("Au, Underground mining and beneficiation at Seabee Gold Operation", "Gold", "CA-SK"),
    "Snow Lake": ("Au and Ag and Cu and Zn, Underground mining and beneficiation at Snow Lake", "Gold, zinc, copper, silver", "CA-MB"),
}


INVENTORIES_metallican_stream = {
#     "Brucejack": ("Underground mining and beneficiation at Brucejack", "Doré", "CA-BC"),
#     "Canadian Malartic": ("Open-pit mining and beneficiation at Canadian Malartic", "Doré", "CA-QC"),
#     "Casa Berardi": ("Open-pit and underground mining and beneficiation at Casa Berardi", "Doré", "CA-QC"),
#     "Detour Lake": ("Open-pit mining and beneficiation at Detour Lake", "Doré", "CA-ON"),
#     "Eagle River": ("Underground mining and beneficiation at Eagle River", "Doré", "CA-ON"),
#     "Éléonore": ("Underground mining and beneficiation at Éléonore", "Doré", "CA-QC"),
#     "Goldex (Au)": ("Underground mining and beneficiation at Goldex", "Doré", "CA-QC"),
#     "Goldex (Cu)": ("Underground mining and beneficiation at Goldex", "Cu, concentrate", "CA-QC"),
#     "Hemlo": ("Open-pit and underground mining and beneficiation at Hemlo (Williams)", "Doré", "CA-ON"),
#     "Kiena": ("Open-pit mining and beneficiation at Kiena", "Doré", "CA-QC"),
#     "Lamaque": ("Underground mining and beneficiation at Lamaque", "Doré", "CA-QC"),
#     "LaRonde (Doré)": ("Underground mining and beneficiation at LaRonde", "Doré", "CA-QC"),
#     "LaRonde (Cu)": ("Underground mining and beneficiation at LaRonde", "Cu concentrate", "CA-QC"),
#     "LaRonde (Zn)": ("Underground mining and beneficiation at LaRonde", "Zn concentrate", "CA-QC"),
#     "Macassa": ("Underground mining and beneficiation at Macassa", "Doré", "CA-ON"),
#     "Meliadine": ("Open-pit and underground mining and beneficiation at Meliadine", "Doré", "CA-NU"),
#     "Musselwhite": ("Underground mining and beneficiation at Musselwhite", "Doré", "CA-ON"),
#     "New Afton": ("Underground mining and beneficiation at New Afton", "Cu concentrate", "CA-BC"),
#     "Rainy River": ("Open-pit and underground mining and beneficiation at Rainy River", "Doré", "CA-ON"),
#     "Red Chris": ("Open-pit mining and beneficiation at Red Chris", "Cu concentrate", "CA-BC"),
#     "Red Lake": ("Underground mining and beneficiation at Red Lake", "Doré", "CA-ON"),
#     "Westwood-Doyon": ("Underground mining and beneficiation at Westwood-Doyon", "Doré", "CA-QC"),
#     "Young-Davidson": ("Underground mining and beneficiation at Young-Davidson", "Doré", "CA-ON"),
#     "Meadowbank complex": ("Open-pit and underground mining at Meadowbank complex", "Doré", "CA-NU"),
#     "Porcupine complex": ("Open-pit and underground mining at Porcupine complex", "Doré", "CA-ON"),
#     "Timmins Operation": ("Underground mining and beneficiation at Timmins Operation", "Doré", "CA-ON"),
#     "Seabee Gold Operation": ("Underground mining and beneficiation at Seabee Gold Operation", "Doré", "CA-SK"),
#     "Snow Lake (Doré)": ("Underground mining and beneficiation at Snow Lake", "Doré", "CA-MB"),
#     "Snow Lake (Cu)": ("Underground mining and beneficiation at Snow Lake", "Cu concentrate", "CA-MB"),
#     "Snow Lake (Zn)": ("Underground mining and beneficiation at Snow Lake", "Zn concentrate", "CA-MB"),

    # Magino
    # Island
    # Fox Complex
    # Highland Valley
    # Mount Milligan
    # Kidd Creek
    # Mount Polley
    # Gibraltar
    # Raglan
    # Long Harbour
    # Thompson
    # Voisey's Bay
    # Nunavut Nickel
    # Cobalt refinery co
    # Keno Hill
    #





}


#
# INVENTORIES_metallican_metal = {
#     "Brucejack (Au)": ("Au, Underground mining and beneficiation at Brucejack", "Au, usable ore", "CA-BC"),
#     "Canadian Malartic (Au)": ("Au and Ag, Open-pit mining and beneficiation at Canadian Malartic", "Au, usable ore", "CA-QC"),
#     "Canadian Malartic (Ag)": ("Au and Ag, Open-pit mining and beneficiation at Canadian Malartic", "Ag, usable ore", "CA-QC"),
#     "Casa Berardi (Au)": ("Au and Ag, Open-pit and underground mining and beneficiation at Casa Berardi", "Au, usable ore", "CA-QC"),
#     "Casa Berardi (Ag)": ("Au and Ag, Open-pit and underground mining and beneficiation at Casa Berardi", "Ag, usable ore", "CA-QC"),
#     "Detour Lake (Au)": ("Au and Ag, Open-pit mining and beneficiation at Detour Lake", "Au, usable ore", "CA-ON"),
#     "Detour Lake (Ag)": ("Au and Ag, Open-pit mining and beneficiation at Detour Lake", "Ag, usable ore", "CA-ON"),
#     "Eagle River (Au)": ("Au, Underground mining and beneficiation at Eagle River", "Au, usable ore", "CA-ON"),
#     "Éléonore (Au)": ("Au and Ag, Underground mining and beneficiation at Éléonore", "Au, usable ore", "CA-QC"),
#     "Éléonore (Ag)": ("Au and Ag, Underground mining and beneficiation at Éléonore", "Ag, usable ore", "CA-QC"),
#     "Goldex (Au)": ("Au and Ag and Zn, Underground mining and beneficiation at Goldex", "Au, usable ore", "CA-QC"),
#     "Goldex (Ag)": ("Au and Ag and Zn, Underground mining and beneficiation at Goldex", "Ag, usable ore", "CA-QC"),
#     "Goldex (Zn)": ("Au and Ag and Zn, Underground mining and beneficiation at Goldex", "Zn, usable ore", "CA-QC"),
#     "Hemlo (Au)": ("Au, Open-pit and underground mining and beneficiation at Hemlo (Williams)", "Au, usable ore", "CA-ON"),
#     "Kiena (Au)": ("Au and Ag, Open-pit mining and beneficiation at Kiena", "Au, usable ore", "CA-QC"),
#     "Kiena (Ag)": ("Au and Ag, Open-pit mining and beneficiation at Kiena", "Ag, usable ore", "CA-QC"),
#     "Lamaque (Au)": ("Au and Ag, Underground mining and beneficiation at Lamaque", "Au, usable ore", "CA-QC"),
#     "Lamaque (Ag)": ("Au and Ag, Underground mining and beneficiation at Lamaque", "Ag, usable ore", "CA-QC"),
#     "LaRonde (Au)": ("Au and Ag and Cu and Zn, Underground mining and beneficiation at LaRonde", "Au, usable ore", "CA-QC"),
#     "LaRonde (Ag)": ("Au and Ag and Cu and Zn, Underground mining and beneficiation at LaRonde", "Ag, usable ore", "CA-QC"),
#     "LaRonde (Cu)": ("Au and Ag and Cu and Zn, Underground mining and beneficiation at LaRonde", "Cu, usable ore", "CA-QC"),
#     "LaRonde (Zn)": ("Au and Ag and Cu and Zn, Underground mining and beneficiation at LaRonde", "Zn, usable ore", "CA-QC"),
#     "Macassa (Au)": ("Au, Underground mining and beneficiation at Macassa", "Au, usable ore", "CA-ON"),
#     "Meliadine (Au)": ("Au and Ag, Open-pit and underground mining and beneficiation at Meliadine", "Au, usable ore", "CA-NU"),
#     "Meliadine (Ag)": ("Au and Ag, Open-pit and underground mining and beneficiation at Meliadine", "Ag, usable ore", "CA-NU"),
#     "Musselwhite (Au)": ("Au, Underground mining and beneficiation at Musselwhite", "Au, usable ore", "CA-ON"),
#     "New Afton (Au)": ("Au and Cu, Underground mining and beneficiation at New Afton", "Au, usable ore", "CA-BC"),
#     "New Afton (Cu)": ("Au and Cu, Underground mining and beneficiation at New Afton", "Cu, usable ore", "CA-BC"),
#     "Rainy River (Au)": ("Au, Open-pit and underground mining and beneficiation at Rainy River", "Au, usable ore", "CA-ON"),
#     "Red Chris (Au)": ("Au and Cu, Open-pit mining and beneficiation at Red Chris", "Au, usable ore", "CA-BC"),
#     "Red Chris (Cu)": ("Au and Cu, Open-pit mining and beneficiation at Red Chris", "Cu, usable ore", "CA-BC"),
#     "Red Lake (Au)": ("Au, Underground mining and beneficiation at Red Lake", "Au, usable ore", "CA-ON"),
#     "Westwood-Doyon (Au)": ("Au and Ag, Underground mining and beneficiation at Westwood-Doyon", "Au, usable ore", "CA-QC"),
#     "Westwood-Doyon (Ag)": ("Au and Ag, Underground mining and beneficiation at Westwood-Doyon", "Ag, usable ore", "CA-QC"),
#     "Young-Davidson (Au)": ("Au, Underground mining and beneficiation at Young-Davidson", "Au, usable ore", "CA-ON"),
#     "Meadowbank complex (Au)": ("Au and Ag, Open-pit and underground mining at Meadowbank complex", "Au, usable ore", "CA-NU"),
#     "Meadowbank complex (Ag)": ("Au and Ag, Open-pit and underground mining at Meadowbank complex", "Ag, usable ore", "CA-NU"),
#     "Porcupine complex (Au)": ("Au, Open-pit and underground mining at Porcupine complex", "Au, usable ore", "CA-ON"),
#     "Timmins Operation (Au)": ("Au and Ag, Underground mining and beneficiation at Timmins Operation", "Au, usable ore", "CA-ON"),
#     "Timmins Operation (Ag)": ("Au and Ag, Underground mining and beneficiation at Timmins Operation", "Ag, usable ore", "CA-ON"),
#     "Seabee Gold Operation (Au)": ("Au, Underground mining and beneficiation at Seabee Gold Operation", "Au, usable ore", "CA-SK"),
#     "Snow Lake (Au)": ("Au and Ag and Cu and Zn, Underground mining and beneficiation at Snow Lake", "Au, usable ore", "CA-MB"),
#     "Snow Lake (Ag)": ("Au and Ag and Cu and Zn, Underground mining and beneficiation at Snow Lake", "Ag, usable ore", "CA-MB"),
#     "Snow Lake (Cu)": ("Au and Ag and Cu and Zn, Underground mining and beneficiation at Snow Lake", "Cu, usable ore", "CA-MB"),
#     "Snow Lake (Zn)": ("Au and Ag and Cu and Zn, Underground mining and beneficiation at Snow Lake", "Zn, usable ore", "CA-MB"),
# }
