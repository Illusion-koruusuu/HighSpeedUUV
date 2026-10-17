"""Map logical rudder/stern commands to HoloOcean Torpedo fin order."""
import numpy as np


def torpedo_fin_commands(rudder, stern, rpm):
    """Return [starboard stern, bottom rudder, port stern, top rudder, rpm]."""
    return np.array([stern, -rudder, -stern, rudder, rpm], dtype=float)
