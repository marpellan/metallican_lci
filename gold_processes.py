# lci_blocks.py

class LCIModule:
    """Base class for all process modules (mining, beneficiation, refining, etc.)."""
    def __init__(self, name, parameters=None, flows=None):
        self.name = name
        self.parameters = parameters or {}  # Dict: {'ore_grade': 0.003, 'electricity_use': 40}
        self.flows = flows or {'technosphere': {}, 'biosphere': {}}  # Dicts of flows

    def set_parameter(self, key, value):
        self.parameters[key] = value

    def add_flow(self, flow_type, flow_name, amount, unit):
        """flow_type: 'technosphere' or 'biosphere'"""
        self.flows[flow_type][flow_name] = {'amount': amount, 'unit': unit}

    def summarize(self):
        """Print a quick summary for checking."""
        print(f"Module: {self.name}")
        print("Parameters:")
        for k, v in self.parameters.items():
            print(f"  {k}: {v}")
        print("Flows:")
        for ftype, flows in self.flows.items():
            print(f"  {ftype}:")
            for name, info in flows.items():
                print(f"    {name}: {info['amount']} {info['unit']}")


class MiningStage(LCIModule):
    def __init__(self, ore_grade, mining_method, electricity_use, diesel_use, cement_use, acid_use, water_use, land_use):
        parameters = {
            'ore_grade': ore_grade,
            'mining_method': mining_method,
            'electricity_use': electricity_use,
            'diesel_use': diesel_use,
            'cement_use': cement_use,
            'acid_use': acid_use,
            'water_use': water_use,
            'land_use': land_use
        }
        super().__init__(name="Mining Stage", parameters=parameters)

class OpenPitMining(MiningStage):
    def __init__(self, ore_grade, electricity_use=50, diesel_use=20, water_use=2.0):
        super().__init__(ore_grade, "open_pit", electricity_use, diesel_use, 0, 0, water_use, 0)
        self.add_flow('technosphere', 'Electricity, medium voltage', electricity_use, 'kWh/t')
        self.add_flow('technosphere', 'Diesel, burned in equipment', diesel_use, 'L/t')
        self.add_flow('biosphere', 'Water use', water_use, 'm3/t')


class BeneficiationStage(LCIModule):
    def __init__(self, process_type, energy_use, water_use):
        parameters = {'process_type': process_type, 'energy_use': energy_use, 'water_use': water_use}
        super().__init__(name="Beneficiation Stage", parameters=parameters)
        self.add_flow('technosphere', 'Electricity, medium voltage', energy_use, 'kWh/t')
        self.add_flow('biosphere', 'Water use', water_use, 'm3/t')



class MiningStage:
    def __init__(self, ore_grade, mining_method, electricity_use, diesel_use,cement_use, acid_use, water_use, land_use):
        self.ore_grade = ore_grade  # e.g., 0.002 (2 g/t)
        self.mining_method = mining_method  # e.g., "open_pit"
        self.electricity_use = electricity_use  # e.g., 50 kWh/tonne (aggregated for the entire mining stage)
        self.diesel_use = diesel_use
        self.cement_use = cement_use
        self.acid_use = acid_use
        self.water_use = water_use  # e.g., 2 m³/tonne
        self.land_use = land_use


class OpenPitMining(MiningStage):
    def __init__(self, ore_grade: float, energy_use: float = None):
        super().__init__(ore_grade, "open_pit")
        self.water_use = 2.0  # Default for open-pit

class UndergroundMining(MiningStage):
    def __init__(self, ore_grade: float, energy_use: float = None):
        super().__init__(ore_grade, "underground")
        self.water_use = 4.0  # Default for underground


class BeneficiationStage:
    def __init__(self, process_type, energy_use, water_use):
        self.process_type = process_type  # e.g., "crushing", "grinding", "flotation"
        self.energy_use = energy_use  # e.g., 30 kWh/tonne (aggregated for beneficiation)
        self.water_use = water_use  # e.g., 1 m³/tonne


class RefiningStage:
    def __init__(self, process_type, energy_use, water_use):
        self.process_type = process_type  # e.g., "smelting", "electrowinning"
        self.energy_use = energy_use  # e.g., 20 kWh/tonne
        self.water_use = water_use  # e.g., 0.5 m³/tonne


# Example site data (aggregated electricity consumption for the entire mining stage)
site_data = {
    "ore_grade": 0.002,
    "mining_method": "open_pit",
    "mining_energy": 50,  # kWh/tonne (total for mining stage)
    "beneficiation_energy": 30,  # kWh/tonne (total for beneficiation stage)
    "refining_energy": 20,  # kWh/tonne (total for refining stage)
}

# Create instances for each stage
mining = MiningStage(
    ore_grade=site_data["ore_grade"],
    mining_method=site_data["mining_method"],
    energy_use=site_data["mining_energy"],
    water_use=None  # Missing data
)

beneficiation = BeneficiationStage(
    process_type="unknown",  # Unknown subprocess (crushing/grinding/flotation)
    energy_use=site_data["beneficiation_energy"],
    water_use=None
)

refining = RefiningStage(
    process_type="smelting",
    energy_use=site_data["refining_energy"],
    water_use=None
)
