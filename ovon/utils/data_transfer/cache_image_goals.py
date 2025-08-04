import argparse
import glob
import json
import os
import random

import habitat
import numpy as np
import torch
from habitat.config import read_write
from habitat_baselines.config.default import get_config

from ovon.models.encoders.siglip_encoder import SigLIPEncoder
from ovon.models.encoders.goat_siglip_encoder import GoatSigLIPEncoder
from open_clip import create_model_from_pretrained, get_tokenizer



class CacheGoals:
    def __init__(
        self,
        config_path: str,
        data_path: str = "",
        split: str = "train",
        output_path: str = "",
        encoder: str = "hf-hub:timm/ViT-B-16-SigLIP-256",
        add_noise: bool = False,
    ) -> None:
        self.device = torch.device("cuda")

        self.config_path = config_path
        self.data_path = data_path
        self.output_path = output_path
        self.split = split
        self.init_visual_encoder(encoder)
        self.encoder_name = encoder
        self.add_noise = add_noise

    def init_visual_encoder(self, encoder="hf-hub:timm/ViT-B-16-SigLIP-256"):
        self.encoder = GoatSigLIPEncoder()
        # self.encoder = SigLIPEncoder()
        # self.encoder = self.encoder.to(device=self.device)
        # self.encoder, _ = create_model_from_pretrained(encoder)

        # if "VC-1" in encoder:
        #     # self.encoder = VC1Encoder(device=self.device)
        # elif "CLIP" in encoder:
        #     self.encoder = CLIPEncoder(device=self.device)
        # else:
            # raise NotImplementedError
        pass

    def apply_noise(self, image):
        mean = 0
        std = random.uniform(0.1, 2.0)
        image = image + np.random.normal(
            loc=mean, scale=std, size=image.shape
        ).astype(np.float32)
        return image.astype(np.uint8)

    def config_env(self, scene):
        config = get_config(self.config_path)
        with read_write(config):
            # config.habitat.dataset.data_path = os.path.join(
            #     self.data_path, f"{self.split}/{self.split}.json.gz"
            # )
            config.habitat.dataset.content_scenes = [scene]

        env = habitat.Env(config=config)
        return env

    def run(self, scene):
        from habitat.tasks.nav.instance_image_nav_task import InstanceImageParameters
        # if os.path.exists(
        #     os.path.join(
        #         self.output_path,
        #         f"{scene}_{self.encoder_name}_goat_embedding.pkl",
        #     )
        # ):
        #     print("Scene already cached: {}".format(scene))
        #     return

        data = {}
        data_goal = {}
        env = self.config_env(scene)
        env.reset()
        goals = env._dataset.goals

        print("Scene reset: {}".format(scene))
        os.makedirs(self.output_path, exist_ok=True)

        print("Add noise: {}".format(self.add_noise))

        for goal_k, goal_vals in goals.items():
            for goal_val in goal_vals:
                goals_meta = []
                for goal_idx, img_goal in enumerate(goal_val["image_goals"]):
                    img_goal = InstanceImageParameters(**img_goal) ## TODO 后续需要依据原本的instace_imagegoal的代码去重构下goat-bench这里的东西，instance_imagegoal都封装为对象了，而这里是字典
                    img = env.task.sensor_suite.sensors[
                        #"instance_imagegoal"
                        "goat_subtask_goal"
                    ]._get_instance_image_goal(img_goal)

                    if self.add_noise:
                        img = self.apply_noise(img)

                    img_embedding = self.encoder.encode_image(img)
                    metadata = dict(
                        hfov=img_goal.hfov,
                        object_id=goal_val.object_id,
                        position=img_goal.position,
                        rotation=img_goal.rotation,
                        goal_id=goal_idx,
                        embedding=img_embedding,
                    )
                    goals_meta.append(metadata)

            scene_id = goal_k.split("_")[0]
            data[f"{goal_k}"] = goals_meta
            data_goal[f"{scene_id}_{goal_val.object_name}"] = goals_meta

        # out_path = os.path.join(
        #     self.output_path, f"{scene}_{self.encoder_name}_iin_embedding.pkl"
        # )
        # save_pickle(data, out_path)

        # out_path = os.path.join(
        #     self.output_path, f"{scene}_{self.encoder_name}_goat_embedding.pkl"
        # )
        # save_pickle(data_goal, out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        #default="config/tasks/instance_imagenav_stretch_hm3d.yaml",
        default="config/tasks/goat_stretch_hm3d.yaml",
    )
    parser.add_argument(
        "--input-path",
        type=str,
        default="",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="./tmp",
    )
    parser.add_argument(
        "--scene",
        type=str,
        default="1S7LAXRdDqk",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
    )
    parser.add_argument(
        "--encoder",
        type=str,
        default="hf-hub:timm/ViT-B-16-SigLIP-256",
    )
    parser.add_argument(
        "--add-noise",
        action="store_true",
        dest="add_noise",
    )
    args = parser.parse_args()

    cache = CacheGoals(
        config_path=args.config,
        data_path=args.input_path,
        split=args.split,
        output_path=args.output_path,
        encoder=args.encoder,
        add_noise=args.add_noise,
    )
    cache.run(args.scene)
