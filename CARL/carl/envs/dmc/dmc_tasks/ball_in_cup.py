# Copyright 2017 The dm_control Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""Ball-in-Cup Domain."""

import collections
from lxml import etree

from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base
from dm_control.suite import common
from dm_control.utils import containers

from carl.utils.types import Context

_DEFAULT_TIME_LIMIT = 20  # (seconds)
_CONTROL_TIMESTEP = .02   # (seconds)


SUITE = containers.TaggedTasks()


def update_physics(xml_string, context):
    """
    Adapts and returns the xml_string of the model with the given context.
    Inspired by
    https://github.com/automl/CARL/blob/main/carl/envs/dmc/dmc_tasks/utils.py
    """

    mjcf = etree.fromstring(xml_string)

    # Update gravity
    option = mjcf.find(".//option")
    if option is None:
        option = etree.Element("option")
        mjcf.append(option)
    gravity_default = option.get("gravity")
    if gravity_default is None:
        gravity = " ".join(["0", "0", str(-context["gravity"])])
    else:
        g = gravity_default.split(" ")
        gravity = " ".join([g[0], g[1], str(-context["gravity"])])
    option.set("gravity", gravity)

    # Update distance cup - ball by moving ball position and tendon length
    tendon = mjcf.find("./tendon/spatial")
    assert tendon is not None
    default_distance = float(tendon.get("range").split(" ")[1])
    delta_distance = default_distance - context["distance"]
    distance = " ".join(["0", str(context["distance"])])
    tendon.set("range", distance)
    bodies = mjcf.findall("./worldbody/body")
    for body in bodies:
        if body.get("name") == "ball":
            p = body.get("pos").split(" ")
            pos = " ".join([p[0], p[1], str(float(p[2]) + delta_distance)])
            body.set("pos", pos)

    xml_string = etree.tostring(mjcf, pretty_print=True)
    return xml_string


def get_model_and_assets():
    """Returns a tuple containing the model XML string and a dict of assets."""
    return common.read_model('ball_in_cup.xml'), common.ASSETS


@SUITE.add('benchmarking', 'easy')
def catch_context(
    context: Context = {},
    time_limit=_DEFAULT_TIME_LIMIT,
    random=None,
    environment_kwargs=None,
):
    """Returns the Ball-in-Cup task."""
    xml_string, assets = get_model_and_assets()
    if context != {}:
        xml_string = update_physics(xml_string=xml_string, context=context)
    physics = Physics.from_xml_string(xml_string, assets)
    task = BallInCup(random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(
        physics, task, time_limit=time_limit, control_timestep=_CONTROL_TIMESTEP,
        **environment_kwargs)


class Physics(mujoco.Physics):
    """Physics with additional features for the Ball-in-Cup domain."""

    def ball_to_target(self):
        """Returns the vector from the ball to the target."""
        target = self.named.data.site_xpos['target', ['x', 'z']]
        ball = self.named.data.xpos['ball', ['x', 'z']]
        return target - ball

    def in_target(self):
        """Returns 1 if the ball is in the target, 0 otherwise."""
        ball_to_target = abs(self.ball_to_target())
        target_size = self.named.model.site_size['target', [0, 2]]
        ball_size = self.named.model.geom_size['ball', 0]
        return float(all(ball_to_target < target_size - ball_size))


class BallInCup(base.Task):
    """The Ball-in-Cup task. Put the ball in the cup."""

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode.

        Args:
          physics: An instance of `Physics`.

        """
        # Find a collision-free random initial position of the ball.
        penetrating = True
        while penetrating:
            # Assign a random ball position.
            physics.named.data.qpos['ball_x'] = self.random.uniform(-.2, .2)
            physics.named.data.qpos['ball_z'] = self.random.uniform(.2, .5)
            # Check for collisions.
            physics.after_reset()
            penetrating = physics.data.ncon > 0
        super().initialize_episode(physics)

    def get_observation(self, physics):
        """Returns an observation of the state."""
        obs = collections.OrderedDict()
        obs['position'] = physics.position()
        obs['velocity'] = physics.velocity()
        return obs

    def get_reward(self, physics):
        """Returns a sparse reward."""
        return physics.in_target()
