import numpy as np

from carl.context.context_space import ContextFeature, UniformFloatContextFeature
from carl.envs.dmc.carl_dmcontrol import CARLDmcEnv


class CARLDmcBallInCupEnv(CARLDmcEnv):
    domain = "ball_in_cup"
    task = "catch_context"
    metadata = {"render_modes": []}

    @staticmethod
    def get_context_features() -> dict[str, ContextFeature]:
        return {
            "gravity": UniformFloatContextFeature(
                "gravity", lower=0.1, upper=np.inf, default_value=9.81
            ),
            "distance": UniformFloatContextFeature(
                "distance", lower=0, upper=np.inf, default_value=0.3
            ),
        }
