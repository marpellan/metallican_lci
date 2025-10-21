"""
lci_modular.py
----------------
Modular, class-based framework to define, parametrize, and complement
LCI data for metal production systems.

Structure:
    - LCIModule: Base class for all process modules
    - MiningStage, BeneficiationStage, RefiningStage: generic process blocks
    - MetalProcess: container assembling multiple modules
    - Example: CopperProcess subclass with metal-specific defaults
"""

from typing import Dict, Optional

# ======================================================
# 1. BASE CLASS
# ======================================================

class LCIModule:
    """
    Base class for any LCI process module (mining, beneficiation, refining, etc.).
    Acts as a structured container for:
      - Parameters (metadata or model drivers)
      - Flows (technosphere and biosphere exchanges)
      - Methods to estimate missing flows

    Attributes
    ----------
    name : str
        Name of the process module (e.g. "Mining Stage")
    parameters : dict
        Key-value pairs defining the main parameters (ore grade, process type, etc.)
    flows : dict
        Contains two nested dicts: {'technosphere': {}, 'biosphere': {}}
        Each flow has {'amount': value, 'unit': str}
    """

    def __init__(self, name: str, parameters: Optional[Dict] = None, flows: Optional[Dict] = None):
        self.name = name
        self.parameters = parameters or {}
        self.flows = flows or {'technosphere': {}, 'biosphere': {}}

    # -------------------------------
    # Basic setters
    # -------------------------------
    def set_parameter(self, key: str, value):
        """Add or update a model parameter."""
        self.parameters[key] = value

    def add_flow(self, flow_type: str, flow_name: str, amount: float, unit: str):
        """Register a flow under 'technosphere' or 'biosphere'."""
        self.flows[flow_type][flow_name] = {'amount': amount, 'unit': unit}

    # -------------------------------
    # Estimation & summary
    # -------------------------------
    def estimate_missing_flows(self):
        """Optional method to be overridden in subclasses."""
        pass

    def summarize(self):
        """Quick textual summary for inspection."""
        print(f"\n=== {self.name} ===")
        print("Parameters:")
        for k, v in self.parameters.items():
            print(f"  {k}: {v}")
        print("Flows:")
        for ftype, flows in self.flows.items():
            print(f"  {ftype}:")
            for name, info in flows.items():
                print(f"    {name}: {info['amount']} {info['unit']}")


# ======================================================
# 2. GENERIC STAGES
# ======================================================

class MiningStage(LCIModule):
    """
    Represents the mining stage (open pit or underground).
    Can estimate missing flows based on ore grade and mining method.
    """

    def __init__(self, ore_grade: float, mining_method: str):
        super().__init__("Mining Stage")
        self.set_parameter("ore_grade", ore_grade)
        self.set_parameter("mining_method", mining_method)
        self.estimate_missing_flows()

    def estimate_missing_flows(self):
        ore_grade = self.parameters.get("ore_grade")
        method = self.parameters.get("mining_method")

        # Parametric relationships (can be refined later)
        if method == "open_pit":
            electricity = 40 * (0.005 / ore_grade) ** 0.2
            diesel = 25 * (0.005 / ore_grade) ** 0.3
            water = 2.0
        else:  # underground
            electricity = 60 * (0.005 / ore_grade) ** 0.2
            diesel = 15 * (0.005 / ore_grade) ** 0.3
            water = 4.0

        # Register flows
        self.add_flow("technosphere", "Electricity, medium voltage", electricity, "kWh/t ore")
        self.add_flow("technosphere", "Diesel, burned in machinery", diesel, "L/t ore")
        self.add_flow("biosphere", "Water use", water, "m3/t ore")


class BeneficiationStage(LCIModule):
    """
    Generic beneficiation (ore processing) module.
    Represents crushing, grinding, flotation, etc.
    """

    def __init__(self, process_type: str, energy_use: Optional[float] = None, water_use: Optional[float] = None):
        super().__init__("Beneficiation Stage")
        self.set_parameter("process_type", process_type)
        self.set_parameter("energy_use", energy_use)
        self.set_parameter("water_use", water_use)
        self.estimate_missing_flows()

    def estimate_missing_flows(self):
        """Set or infer default flows if missing."""
        energy = self.parameters.get("energy_use") or 30.0  # default kWh/t
        water = self.parameters.get("water_use") or 1.0     # default m³/t

        self.add_flow("technosphere", "Electricity, medium voltage", energy, "kWh/t ore")
        self.add_flow("biosphere", "Water use", water, "m3/t ore")


class RefiningStage(LCIModule):
    """
    Generic refining (smelting, electrowinning, etc.) module.
    """

    def __init__(self, process_type: str, energy_use: Optional[float] = None, water_use: Optional[float] = None):
        super().__init__("Refining Stage")
        self.set_parameter("process_type", process_type)
        self.set_parameter("energy_use", energy_use)
        self.set_parameter("water_use", water_use)
        self.estimate_missing_flows()

    def estimate_missing_flows(self):
        energy = self.parameters.get("energy_use") or 20.0
        water = self.parameters.get("water_use") or 0.5

        self.add_flow("technosphere", "Electricity, medium voltage", energy, "kWh/t concentrate")
        self.add_flow("biosphere", "Water use", water, "m3/t concentrate")


# ======================================================
# 3. COMPOSITION OF STAGES
# ======================================================

class MetalProcess:
    """
    Assembles several process modules (mining, beneficiation, refining)
    into a complete life cycle inventory for one metal.
    """

    def __init__(self, metal_name: str):
        self.name = metal_name
        self.modules = []

    def add_module(self, module: LCIModule):
        """Add a stage (MiningStage, BeneficiationStage, etc.)"""
        self.modules.append(module)

    def summarize(self):
        """Display all included modules and flows."""
        print(f"\n### Metal Process: {self.name} ###")
        for m in self.modules:
            m.summarize()

    def get_combined_flows(self):
        """Merge all flows from each module into a single dictionary."""
        combined = {'technosphere': {}, 'biosphere': {}}
        for m in self.modules:
            for ftype in m.flows:
                for fname, fdata in m.flows[ftype].items():
                    # Sum if repeated
                    combined[ftype][fname] = combined[ftype].get(fname, {'amount': 0, 'unit': fdata['unit']})
                    combined[ftype][fname]['amount'] += fdata['amount']
        return combined


# ======================================================
# 4. EXAMPLE METAL-SPECIFIC IMPLEMENTATION
# ======================================================

class CopperProcess(MetalProcess):
    """
    Example specialization for copper.
    Defines a default sequence of process modules with copper-specific logic.
    """

    def __init__(self, ore_grade=0.005, mining_method="open_pit"):
        super().__init__("Copper")

        mining = MiningStage(ore_grade, mining_method)
        beneficiation = BeneficiationStage(process_type="flotation")
        refining = RefiningStage(process_type="smelting")

        self.add_module(mining)
        self.add_module(beneficiation)
        self.add_module(refining)
