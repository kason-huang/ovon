#!/usr/bin/env python3

import json
import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence

import attr
from habitat.core.registry import registry
from habitat.core.simulator import AgentState
from habitat.core.utils import DatasetFloatJSONEncoder
from habitat.datasets.pointnav.pointnav_dataset import (
    CONTENT_SCENES_PATH_FIELD,
    DEFAULT_SCENE_PATH_PREFIX,
    PointNavDatasetV1,
)
from habitat.tasks.nav.instance_image_nav_task import (  # InstanceImageGoalNavEpisode,
    InstanceImageGoal,
    InstanceImageParameters,
)
from habitat.tasks.nav.object_nav_task import ObjectGoal, ObjectViewLocation
from habitat.core.utils import not_none_validator

from ovon.task.goat_task import GoatEpisode

if TYPE_CHECKING:
    from omegaconf import DictConfig


@attr.s(auto_attribs=True, kw_only=True)
class GoatGoal(ObjectGoal):
    """An instance image goal is an ObjectGoal that also contains a collection
    of InstanceImageParameters.

    Args:
        image_goals: a list of camera parameters each used to generate an
        image goal.
    """

    image_goals: List[InstanceImageParameters] = attr.ib(
        default=None, validator=not_none_validator
    )
    object_surface_area: Optional[float] = None
    children_object_categories: Optional[List[str]] = []
    lang_desc: Optional[str] = None


@registry.register_dataset(name="Goat-v1")
class GoatDatasetV1(PointNavDatasetV1):
    r"""
    Class inherited from PointNavDataset that loads GOAT dataset.
    """
    episodes: List[GoatEpisode] = []  # type: ignore
    content_scenes_path: str = "{data_path}/content/{scene}.json.gz"
    goals_by_category: Dict[str, Sequence[ObjectGoal]]

    def to_json(self) -> str:
        for i in range(len(self.episodes)):
            self.episodes[i].goals = []

        result = DatasetFloatJSONEncoder().encode(self)

        for i in range(len(self.episodes)):
            goals = self.goals_by_category[self.episodes[i].goals_key]
            if not isinstance(goals, list):
                goals = list(goals)
            self.episodes[i].goals = goals

        return result

    def __init__(self, config: Optional["DictConfig"] = None) -> None:
        self.goals_by_category = {}
        super().__init__(config)
        self.episodes = list(self.episodes)

    @staticmethod
    def __deserialize_objectnav_goal(
        serialized_goal: Dict[str, Any]
    ) -> ObjectGoal:
        g = ObjectGoal(**serialized_goal)

        for vidx, view in enumerate(g.view_points):
            view_location = ObjectViewLocation(**view)  # type: ignore
            view_location.agent_state = AgentState(**view_location.agent_state)  # type: ignore
            g.view_points[vidx] = view_location

        return g

    @staticmethod
    def __deserialize_languagenav_goal(
        serialized_goal: Dict[str, Any]
    ) -> ObjectGoal:
        if serialized_goal.get("children_object_categories") is not None:
            del serialized_goal["children_object_categories"]

        g = ObjectGoal(**serialized_goal)

        for vidx, view in enumerate(g.view_points):
            view_location = OVONObjectViewLocation(**view)  # type: ignore
            view_location.agent_state = AgentState(**view_location.agent_state)  # type: ignore
            g.view_points[vidx] = view_location

        return g

    @staticmethod
    def __deserialize_imagenav_goal(
        serialized_goal: Dict[str, Any]
    ) -> InstanceImageGoal:
        g = InstanceImageGoal(**serialized_goal)

        for vidx, view in enumerate(g.view_points):
            view_location = ObjectViewLocation(**view)  # type: ignore[arg-type]
            view_location.agent_state = AgentState(**view_location.agent_state)  # type: ignore[arg-type]
            g.view_points[vidx] = view_location

        for iidx, params in enumerate(g.image_goals):
            g.image_goals[iidx] = InstanceImageParameters(**params)  # type: ignore[arg-type]

        return g
    
    @staticmethod
    def __deserialize_goal(
        serialized_goal: Dict[str, Any]
    ) -> GoatGoal:
        g = GoatGoal(**serialized_goal)

        for vidx, view in enumerate(g.view_points):
            view_location = ObjectViewLocation(**view)  # type: ignore[arg-type]
            view_location.agent_state = AgentState(**view_location.agent_state)  # type: ignore[arg-type]
            g.view_points[vidx] = view_location

        for iidx, params in enumerate(g.image_goals):
            g.image_goals[iidx] = InstanceImageParameters(**params)  # type: ignore[arg-type]

        return g

    def from_json(
        self, json_str: str, scenes_dir: Optional[str] = None
    ) -> None:
        deserialized = json.loads(json_str)
        if CONTENT_SCENES_PATH_FIELD in deserialized:
            self.content_scenes_path = deserialized[CONTENT_SCENES_PATH_FIELD]

        if len(deserialized["episodes"]) == 0:
            return
        
        for k, v in deserialized["goals_by_category"].items():
            self.goals_by_category[k] = [self.__deserialize_goal(g) for g in v]

        num_filtered_eps = 0

        for i, composite_episode in enumerate(deserialized["episodes"]):
            composite_episode["goals"] = []
            composite_episode = GoatEpisode(**composite_episode)

            composite_episode.episode_id = str(i)

            if scenes_dir is not None:
                if composite_episode.scene_id.startswith(
                    DEFAULT_SCENE_PATH_PREFIX
                ):
                    composite_episode.scene_id = composite_episode.scene_id[
                        len(DEFAULT_SCENE_PATH_PREFIX) :
                    ]

                composite_episode.scene_id = os.path.join(
                    scenes_dir, "", composite_episode.scene_id
                )

            composite_episode.goals = []

            for idx, goal in enumerate(composite_episode.tasks):
                goal_type = goal[1]
                goal_category = goal[0]
                goal_inst_id = goal[2]

                # goals_by_category里面累积了所有scene的东西
                dset_same_cat_goals = self.goals_by_category[composite_episode.goals_key_by_idx(idx)]
                # 处理子类型
                if len(dset_same_cat_goals) == 0:
                    print(f"goals_key not exist {composite_episode.goals_key_by_idx(idx)}")
                    exit(-1)

                children_categories = dset_same_cat_goals[0].children_object_categories

                for child_category in children_categories:
                    goal_key = "{}_{}".format(
                        composite_episode.scene_id.split("/")[-1],
                        child_category,
                    )
                    if goal_key not in self.goals_by_category:
                        continue
                    dset_same_cat_goals.extend(self.goals_by_category[goal_key])

                if goal_type == "object":
                    composite_episode.goals.append(dset_same_cat_goals)
                else:
                    # 这里用列表是是为与object对齐，object里面的也是一个列表
                    goal_inst = [
                        x
                        for x in dset_same_cat_goals
                        if x.object_id == goal_inst_id
                    ]
                    composite_episode.goals.append(goal_inst)
                
            self.episodes.append(composite_episode)  # type: ignore [attr-defined]
