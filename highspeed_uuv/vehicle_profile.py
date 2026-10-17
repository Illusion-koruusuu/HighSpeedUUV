"""Shared HoloOcean Torpedo actuator profiles for training and replay."""

STANDARD_PROFILE = "standard"
BOOSTED_PROFILE = "boosted"
DEFAULT_PROFILE = BOOSTED_PROFILE
FIN_MAPPING_VERSION = 2

# The boosted setting is an extrapolation of HoloOcean's Fossen Torpedo model,
# not a measured REMUS100 performance specification.  It retains the original
# hull, fin and propeller equations while raising only the RPM ceiling.
ACTUATOR_OVERRIDES = {
    STANDARD_PROFILE: {},
    BOOSTED_PROFILE: {"nMax": 3000},
}


def actuator_overrides(profile):
    try:
        return ACTUATOR_OVERRIDES[profile].copy()
    except KeyError as exc:
        raise ValueError(f"unknown HoloOcean vehicle profile: {profile}") from exc
