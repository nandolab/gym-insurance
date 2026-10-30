"""Versioned research environments; legacy v0 is loaded only when requested."""
from gymnasium.envs.registration import register, registry

for name, entry in {
    "InsurEnv-v0": "gym_insurance.envs.insurenv:InsurEnv",
    "InsurEnv-v1": "gym_insurance.v1:InsurEnv",
}.items():
    if name not in registry:
        register(id=name, entry_point=entry)

from .v1 import InsurEnv

__version__ = "0.2.0.dev0"
