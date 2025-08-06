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
from ovon.utils.utils import save_image, save_pickle, load_pickle, load_dataset
from tqdm import tqdm

def load_categories_from_dataset(path):
    if not os.path.exists(path):
        print(f"Warning: path '{path}' does not exist.")
        return []
    
    files = glob.glob(os.path.join(path, "*json.gz"))

    categories = []
    for f in tqdm(files):
        dataset = load_dataset(f)
        for goal_key in dataset["goals_by_category"].keys():
            categories.append(goal_key.split("_")[1])
    return list(set(categories))

class CacheGoals:
    def __init__(
        self,
        config_path: str,
        dataset_path: str = "",
        split: str = "train",
        output_path: str = "",
        encoder: str = "siglip",
        add_noise: bool = False,
    ) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.config_path = config_path
        self.dataset_path = dataset_path
        self.output_path = output_path
        self.split = split
        self.encoder_name = encoder
        self.add_noise = add_noise

        self.init_encoder()

    def init_encoder(self):
        if self.encoder_name == "siglip":
            self.encoder = GoatSigLIPEncoder(self.device)
            self.encoder.eval()
        else:
            raise NotImplemented("invalid encoder name")
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
    
    
    def run_cache_object_goals(self):
        dataset_path = f"{self.dataset_path}/train/content"
        goal_categories = load_categories_from_dataset(dataset_path)
        dataset_path = f"{self.dataset_path}/val_seen/content"
        val_seen_categories = load_categories_from_dataset(
            dataset_path
        )
        dataset_path = f"{self.dataset_path}/val_seen_synonyms/content"
        val_seen_synoyms_categories = load_categories_from_dataset(
            dataset_path
        )
        dataset_path = f"{self.dataset_path}/val_unseen/content"
        val_unseen_categories = load_categories_from_dataset(
            dataset_path
        )

        # Print the first 5 categories of each split
        print("Total train categories num: {}".format(len(goal_categories)))
        print("First 5 categories:")
        print("goal_categories: {}".format(goal_categories[:5]))
        print("val_seen_categories num : {}".format(val_seen_categories[:5]))
        print("val_seen_synoyms_categories num: {}".format(len(val_seen_synoyms_categories)))
        print("val_unseen_categories num: {}".format(len(val_unseen_categories)))

        goal_categories.extend(val_seen_categories)
        goal_categories.extend(val_seen_synoyms_categories)
        goal_categories.extend(val_unseen_categories)

        print("Total goal categories: {}".format(len(goal_categories)))
        print(
            "Train categories: {}, Val seen categories: {}, Val unseen easy categories: {},"
            " Val unseen hard categories: {}".format(
                len(goal_categories),
                len(val_seen_categories),
                len(val_seen_synoyms_categories),
                len(val_unseen_categories),
            )
        )

        text_embedding = self.encoder.encode_text(goal_categories)
        print(text_embedding.shape)

        output_path = f"./tmp/category_name_{self.encoder_name}_embedding.pkl"
        output = {}
        for goal_category, embedding in zip(goal_categories, text_embedding):
            output[goal_category] = embedding.detach().cpu().numpy()
        save_pickle(output, output_path)


    def run_cache_image_goals(self, scene):
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
                if "image_goals" not in goal_val or not goal_val["image_goals"]:
                    continue
                for goal_idx, img_goal in enumerate(goal_val["image_goals"]):
                    img_goal = InstanceImageParameters(**img_goal) ## TODO 后续需要依据原本的instace_imagegoal的代码去重构下goat-bench这里的东西，instance_imagegoal都封装为对象了，而这里是字典
                    img = env.task.sensor_suite.sensors[
                        #"instance_imagegoal"
                        "goat_subtask_raw_goal"
                    ]._get_instance_image_goal(img_goal)

                    if self.add_noise:
                        img = self.apply_noise(img)

                    img_embedding = self.encoder.encode_image(img)
                    # --- for debug -------
                    text = [f"a photo of {goal_val['object_category']}", "a photo of dog"]
                    text_embedding = self.encoder.encode_text(text)

                    image_features = img_embedding / img_embedding.norm(dim=-1, keepdim=True)
                    text_features = text_embedding / text_embedding.norm(dim=-1, keepdim=True)
                    similarity = (image_features @ text_features.T).squeeze(0)  # shape: (num_texts,)
                    print(f"category {goal_val['object_category']} similarity: {similarity}")
                    # ----- for debug end

                    img_embedding =  img_embedding.squeeze().detach().cpu().numpy()
                    metadata = dict(
                        hfov=img_goal.hfov,
                        object_id=goal_val["object_id"],
                        position=img_goal.position,
                        rotation=img_goal.rotation,
                        goal_id=goal_idx,
                        embedding=img_embedding,
                    )
                    goals_meta.append(metadata)


                scene_id = goal_k.split("_")[0]
                object_id = goal_val["object_id"]
                data_goal[f"{scene_id}_{object_id}"] = goals_meta

        out_path = os.path.join(
            self.output_path, f"{scene}_{self.encoder_name}_embedding.pkl"
        )
        save_pickle(data_goal, out_path)

        # out_path = os.path.join(
        #     self.output_path, f"{scene}_{self.encoder_name}_goat_embedding.pkl"
        # )
        # save_pickle(data_goal, out_path)
    
    def load(self, scene):
        out_path = os.path.join(
            self.output_path, f"{scene}_{self.encoder_name}_embedding.pkl"
        )
        data = load_pickle(out_path)
        print(len(data))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        #default="config/tasks/instance_imagenav_stretch_hm3d.yaml",
        default="config/tasks/goat_stretch_hm3d.yaml",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="./data/datasets/ovon/hm3d/v2",
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
        default="siglip",
    )
    parser.add_argument(
        "--add-noise",
        action="store_true",
        dest="add_noise",
    )
    args = parser.parse_args()

    cache = CacheGoals(
        config_path=args.config,
        dataset_path=args.dataset_path, # 这个dataset_path只是给object goal使用的，因为image goal需要habitat仿真器配合所以用的dataset path是配置文件的
        split=args.split,
        output_path=args.output_path,
        encoder=args.encoder,
        add_noise=args.add_noise,
    )
    # cache.run(args.scene)
    # cache.load(args.scene)
    cache.run_cache_object_goals()
