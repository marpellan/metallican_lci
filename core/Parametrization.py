import math
from dataclasses import dataclass


# ======================================================
# From Argonne
# ======================================================

@dataclass
class OreEnergyModel:
    metal: str
    a: float
    b: float
    c: float
    grade_ref: float  # %
    diesel_share_mining: float
    elec_share_mining: float
    elec_share_benef: float
    mining_share: float
    benef_share: float

    def process_energy(self, grade_pct):
        """Return process energy (MJ/kg metal) as function of ore grade (%)."""
        G = grade_pct
        return self.a + self.b / G - self.c * G

# Example (Nickel sulfide defaults from GREET reference):
nickel_model = OreEnergyModel(
    metal="Ni_sulfide",
    a=0, b=0, c=0,         # placeholder until GREET constants are extracted
    grade_ref=2.05,
    diesel_share_mining=0.304,
    elec_share_mining=0.696,
    elec_share_benef=1.0,
    mining_share=0.5172,
    benef_share=0.4828
)
