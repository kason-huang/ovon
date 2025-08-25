from dataclasses import dataclass
from typing import Any

from habitat import EmbodiedTask, Measure, Simulator, registry
from habitat.config.default_structured_configs import MeasurementConfig
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig
from ovon.task.goat_task import GoatEpisode

@registry.register_measure
class ObjectTaskCount(Measure):
    """The measure calculates an angle towards the goal. Note: this measure is
    only valid for single goal tasks (e.g., ImageNav)
    """

    cls_uuid: str = "object_task_count"

    def __init__(self, *args, **kwargs):
        super().__init__()
        self._metric = 0               # 对外暴露的标量
        self._count = 0        # 内部累计计数（跨 episode 保持）
    
    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
         return self.cls_uuid

    def reset_metric(self, episode:GoatEpisode, *args, **kwargs):
        # 每次新 episode 开始时调用
        if episode.tasks[episode.active_subtask_idx][1] == 'object':
            self._count += 1
        self._metric = self._count

    def update_metric(self, task, *args, **kwargs):
        # EpisodeCount 在 reset 时就定值，本函数可为空
        pass

    @property
    def metric(self):
        return self._metric