# goat_goal_helper.py
import numpy as np
from gym import spaces

import habitat_sim
from habitat.utils.geometry_utils import quaternion_from_coeff
from habitat_sim import bindings as hsim
from habitat_sim.agent.agent import AgentState, SixDOFPose
from habitat.tasks.nav.instance_image_nav_task import InstanceImageParameters
from ovon.utils.utils import load_pickle

class GoatGoalHelper:
    def __init__(self, sim, config):
        self._sim = sim
        self.cfg = config
        # 你之前的 cache / 维度检查
        self.object_cache = load_pickle(config.object_cache)
        k0 = next(iter(self.object_cache.keys()))
        self.embed_dim = int(self.object_cache[k0].shape[0])
        for v in self.object_cache.values():
            assert v.ndim == 1 and v.shape[0] == self.embed_dim
        #self.image_shape = (256, 256, 3)  # 或从 config 读
        self.image_shape = (512, 512, 3)  # 这个是进来的还是出去的呢？
        self._image_cache = {}  # (scene_id, ep_id, sub_idx, img_idx) -> np.uint8(H,W,3)

    # —— 渲染 image goal：和你原来的一样，只是搬到 helper 里 ——
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

    def _render_image_goal(self, img_params):
        sensor_uuid = "goat_goal_tmp_sensor"
        self._add_sensor(img_params, sensor_uuid)
        self._sim._sensors[sensor_uuid].draw_observation()
        img = self._sim._sensors[sensor_uuid].get_observation()[:, :, :3]
        self._remove_sensor(sensor_uuid)
        return img

    # —— 关键接口：返回当前 subtask 的三件东西（image/object/mask），并做缓存 ——
    def get_goal_pack(self, episode, task):
        # 这里的 task/episode 索引按你已有的数据结构取
        sub_idx = task.active_subtask_idx
        ttype = episode.tasks[sub_idx][1]  # "object" or "image"

        if ttype == "object":
            category = episode.tasks[sub_idx][0]
            emb = self.object_cache.get(category, None)
            if emb is None:
                print(f"[GoatGoal] Missing category: {category}")
                emb = np.zeros((self.embed_dim,), np.float32)
            else:
                emb = emb.astype(np.float32, copy=False)
            return {
                "image": None,
                "object": emb,
                "mask": np.array([0, 1], np.int8),   # [has_image, has_object]
            }

        elif ttype == "image":
            img_idx = episode.tasks[sub_idx][3]
            key = (episode.scene_id, episode.episode_id, sub_idx, img_idx)
            img = self._image_cache.get(key)
            if img is None:
                img_params = episode.goals[sub_idx][0].image_goals[img_idx]
                img = self._render_image_goal(img_params)
                # 可选：一致性检查尺寸
                if img.shape[:2] != self.image_shape[:2]:
                    # 需要的话，这里 resize 到 self.image_shape
                    pass
                self._image_cache[key] = img
            return {
                "image": img,
                "object": None,
                "mask": np.array([1, 0], np.int8),
            }

        else:
            # 兜底：不应该发生
            return {
                "image": np.zeros(self.image_shape, np.uint8),
                "object": np.zeros((self.embed_dim,), np.float32),
                "mask": np.array([0, 0], np.int8),
            }
