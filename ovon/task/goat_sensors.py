import hashlib
import os
import random
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
from gym import spaces

import habitat_sim
from habitat.core.embodied_task import EmbodiedTask
from habitat.core.registry import registry
from habitat.core.simulator import RGBSensor, Sensor, SensorTypes, Simulator, VisualObservation
from habitat.core.utils import try_cv2_import
from habitat.tasks.nav.nav import NavigationEpisode
from habitat.tasks.nav.instance_image_nav_task import InstanceImageParameters

from habitat.utils.geometry_utils import quaternion_from_coeff
from habitat_sim import bindings as hsim
from habitat_sim.agent.agent import AgentState, SixDOFPose
from ovon.utils.utils import load_pickle

@registry.register_sensor
class GoatGoalSensor(Sensor):
    r"""A sensor for Goat goals with cached embedding type"""

    cls_uuid: str = "goat_subtask_goal"

    def __init__(
        self,
        sim,
        config: "DictConfig",
        **kwargs: Any,
    ):
        self._sim = sim
        self.image_cache_base_dir = config.image_cache
        self.image_encoder = config.image_cache_encoder
        self.image_cache = None
        self.object_cache = load_pickle(config.object_cache)
        self._current_scene_id = ""
        self._current_episode_id = ""
        self._current_episode_image_goal = None

        k = list(self.object_cache.keys())[0]
        self._embed_dim = self.object_cache[k].shape[0] # the _embed_dim == 768
        for v in self.object_cache.values():
            assert self._embed_dim == v.shape[0] and v.ndim == 1
            
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.SEMANTIC

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            #low=-np.inf, high=np.inf, shape=(1024,), dtype=np.float32
            low=-np.inf, high=np.inf, shape=(self._embed_dim,), dtype=np.float32
        )

    def get_observation(
        self,
        observations,
        *args: Any,
        episode: Any,
        task: Any,
        **kwargs: Any,
    ) -> np.ndarray:
        if self._current_scene_id != episode.scene_id:
            self._current_scene_id = episode.scene_id
   
        # we only calcuate the first sub task now
        task_type = episode.tasks[task.active_subtask_idx][1]
        if task_type == "object":
            category = episode.tasks[task.active_subtask_idx][0]
            if category not in self.object_cache:
                print("Missing category: {}".format(category))
            return self.object_cache[category]
            # return {
                # "type": task_type,
                # "value": self.object_cache[category]
            # }
        elif task_type == "image":
            scene_id = episode.scene_id.split("/")[-1].split(".")[0]
            self.image_cache = load_pickle(
                os.path.join(
                    self.image_cache_base_dir,
                    f"{scene_id}_{self.image_encoder}_embedding.pkl",
                )
            )

            instance_id = episode.tasks[task.active_subtask_idx][2]
            img_idx = episode.tasks[task.active_subtask_idx][3]
            #scene_id = episode.scene_id.split("/")[-1].split(".")[0]
            scene_glb = episode.scene_id.split("/")[-1]
            
            uuid = "{}_{}".format(scene_glb, instance_id)

            output_embedding = self.image_cache[uuid][img_idx]["embedding"]
            return output_embedding
            # return {
                # "type": task_type,
                # "value": output_embedding
            # }


@registry.register_sensor
class GoatModalTypeSensor(Sensor):
    cls_uuid: str = "goat_modal_type"

    mapping = {
        "image": 2,
        "object": 1,
    }

    def _get_observation_space(self, *args, **kwargs):
        return spaces.Discrete(1)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid
    
    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.TENSOR
    
    def get_observation(
        self,
        observations,
        *args: Any,
        episode: Any,
        task: Any,
        **kwargs: Any,
    ) -> np.ndarray:
        task_type = episode.tasks[task.active_subtask_idx][1]
        return self.mapping[task_type]

@registry.register_sensor
class GoatRawGoalSensor(Sensor):
    r"""A sensor for Goat goals"""

    cls_uuid: str = "goat_subtask_raw_goal"

    def __init__(
        self,
        sim,
        config: "DictConfig",
        **kwargs: Any,
    ):
        self._sim = sim
        self.image_cache_base_dir = config.image_cache
        self.image_encoder = config.image_cache_encoder
        self.image_cache = None
        # self.language_cache = load_pickle(config.language_cache)
        # self.object_cache = load_pickle(config.object_cache)
        self._current_scene_id = ""
        self._current_episode_id = ""
        self._current_episode_image_goal = None
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.SEMANTIC

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=-np.inf, high=np.inf, shape=(1024,), dtype=np.float32
        )
    
    def _add_sensor(
        self, img_params: InstanceImageParameters, sensor_uuid: str
    ) -> None:
        spec = habitat_sim.CameraSensorSpec()
        spec.uuid = sensor_uuid
        spec.sensor_type = habitat_sim.SensorType.COLOR
        spec.resolution = img_params.image_dimensions
        spec.hfov = img_params.hfov
        spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
        self._sim.add_sensor(spec)

        agent = self._sim.get_agent(0)
        agent_state = agent.get_state()
        agent.set_state(
            AgentState(
                position=agent_state.position,
                rotation=agent_state.rotation,
                sensor_states={
                    **agent_state.sensor_states,
                    sensor_uuid: SixDOFPose(
                        position=np.array(img_params.position),
                        rotation=quaternion_from_coeff(img_params.rotation),
                    ),
                },
            ),
            infer_sensor_states=False,
        )

    def _remove_sensor(self, sensor_uuid: str) -> None:
        agent = self._sim.get_agent(0)
        del self._sim._sensors[sensor_uuid]
        hsim.SensorFactory.delete_subtree_sensor(agent.scene_node, sensor_uuid)
        del agent._sensors[sensor_uuid]
        agent.agent_config.sensor_specifications = [
            s
            for s in agent.agent_config.sensor_specifications
            if s.uuid != sensor_uuid
        ]

    def _get_instance_image_goal(
        self, img_params: InstanceImageParameters
    ) -> VisualObservation:
        """To render the instance image goal, a temporary HabitatSim sensor is
        created with the specified InstanceImageParameters. This sensor renders
        the image and is then removed.
        """
        sensor_uuid = f"{self.cls_uuid}_sensor"
        self._add_sensor(img_params, sensor_uuid)

        self._sim._sensors[sensor_uuid].draw_observation()
        img = self._sim._sensors[sensor_uuid].get_observation()[:, :, :3]

        self._remove_sensor(sensor_uuid)
        return img
    
    def get_observation(
        self,
        observations,
        *args: Any,
        episode: Any,
        task: Any,
        **kwargs: Any,
    ) -> np.ndarray:
        episode_id = f"{episode.scene_id}_{episode.episode_id}"

        if self._current_scene_id != episode.scene_id:
            self._current_scene_id = episode.scene_id
            scene_id = episode.scene_id.split("/")[-1].split(".")[0]
            # self.image_cache = load_pickle(
            #     os.path.join(
            #         self.image_cache_base_dir,
            #         f"{scene_id}_{self.image_encoder}_embedding.pkl",
            #     )
            # )
        
        task_type = "none"
        # we only calcuate the first sub task now
        task_type = episode.tasks[task.active_subtask_idx][1]
        if task_type == "object":
            self.category = episode.tasks[task.active_subtask_idx][0]
            return self.category
            # return {
                # "type": task_type,
                # "value": self.category
            # }
        elif task_type == "image":
            img_idx = episode.tasks[task.active_subtask_idx][3]
            # 这里的0表示的就是object_id对应的goal，所以其实应该就只有一个才对，后续可以把这个数组给去掉
            img_goal  = episode.goals[task.active_subtask_idx][0]['image_goals'][img_idx]
            img_goal = InstanceImageParameters(**img_goal)
            self._current_image_goal = self._get_instance_image_goal(img_goal)
            return self._current_image_goal
            # return {
                # "type": task_type,
                # "value": self._current_image_goal
            # }

        # output_embedding = np.zeros((1024,), dtype=np.float32)

        #TODO 我们需要根据image_cached的那个代码用类似ClipObjectCache或者说Instance_ImageGoal那些方法把这些想要给返回回去
        # task_type = "none"
        # if task.active_subtask_idx < len(episode.tasks):
        #     if episode.tasks[task.active_subtask_idx][1] == "object":
        #         category = episode.tasks[task.active_subtask_idx][0]
        #         output_embedding = self.object_cache[category]
        #         task_type = "object"
        #     elif episode.tasks[task.active_subtask_idx][1] == "description":
        #         instance_id = episode.tasks[task.active_subtask_idx][2]
        #         goal = [
        #             g
        #             for g in episode.goals[task.active_subtask_idx]
        #             if g["object_id"] == instance_id
        #         ]
        #         uuid = goal[0]["lang_desc"].lower()
        #         output_embedding = self.language_cache[uuid]
        #         task_type = "lang"
        #     elif episode.tasks[task.active_subtask_idx][1] == "image":
        #         instance_id = episode.tasks[task.active_subtask_idx][2]
        #         curent_task = episode.tasks[task.active_subtask_idx]
        #         scene_id = episode.scene_id.split("/")[-1].split(".")[0]

        #         uuid = "{}_{}".format(scene_id, instance_id)

        #         output_embedding = self.image_cache[
        #             "{}_{}".format(scene_id, instance_id)
        #         ][curent_task[-1]]["embedding"]
        #         task_type = "image"
        #     else:
        #         raise NotImplementedError