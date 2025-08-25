import argparse
import glob
import json
import os
import time
import random
from pathlib import Path

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
from contextlib import contextmanager

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
        skip_if_exists: bool = True,
    ) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.config_path = config_path
        self.dataset_path = dataset_path
        self.output_path = output_path
        self.skip_if_exists = skip_if_exists
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

    @contextmanager
    def _safe_env(self, scene):
        env = self.config_env(scene)
        try:
            env.reset()
            yield env
        finally:
            try:
                env.close()
            except Exception:
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
        dataset_path = f"{self.dataset_path}/ovon_val_seen/content"
        ovon_val_seen_categories = load_categories_from_dataset(
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
        goal_categories = list(set(goal_categories))
        print("Total goal categories (only goat): {}".format(len(goal_categories)))

        goal_categories.extend(ovon_val_seen_categories)
        goal_categories = list(set(goal_categories))
        print("Total goal categories (include ovon seen): {}".format(len(goal_categories)))

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

        output_path = f"./tmp/ovon/category_name_{self.encoder_name}_embedding.pkl"
        output = {}
        for goal_category, embedding in zip(goal_categories, text_embedding):
            output[goal_category] = embedding.detach().cpu().numpy()
        save_pickle(output, output_path)



    def run_cache_image_goals(self, scene):
        # --- 计时：总时长 ---
        t_total_start = time.perf_counter()

        from habitat.tasks.nav.instance_image_nav_task import InstanceImageParameters
        if self.skip_if_exists and os.path.exists(
            os.path.join(self.output_path, f"{scene}_{self.encoder_name}_embedding.pkl")
        ):
            print("Scene already cached: {}".format(scene))
            return
        
        t_env_start = time.perf_counter()
        data_goal = {}
        env = self.config_env(scene)
        env.reset()
        goals = env._dataset.goals_by_category
        t_env = time.perf_counter() - t_env_start

        print("Scene reset: {}".format(scene))
        os.makedirs(self.output_path, exist_ok=True)

        print("Add noise: {}".format(self.add_noise))

        n_images = 0
        t_noise = 0.0
        t_sensor = 0.0
        t_encode = 0.0
        t_debug = 0.0   # 你用于 text 相似度的调试耗时
        t_save = 0.0

        for goal_k, goal_vals in goals.items():
            for goal_val in goal_vals:
                goals_meta = []
                if len(goal_val.image_goals) == 0:
                    continue
                for goal_idx, img_goal in enumerate(goal_val.image_goals):
                    t_sensor_start = time.perf_counter()
                    img = env.task.sensor_suite.sensors[
                        #"instance_imagegoal"
                        "goat_subtask_raw_goal"
                    ]._get_instance_image_goal(img_goal)
                    t_sensor += time.perf_counter() - t_sensor_start

                    if self.add_noise:
                        img = self.apply_noise(img)

                    t1 = time.perf_counter()
                    img_embedding = self.encoder.encode_image(img)
                    t_encode += time.perf_counter() - t1
                    n_images += 1

                    # # --- for debug -------
                    # t2 = time.perf_counter()
                    # text = [f"a photo of {goal_val.object_category}", "a photo of dog"]
                    # text_embedding = self.encoder.encode_text(text)

                    # image_features = img_embedding / img_embedding.norm(dim=-1, keepdim=True)
                    # text_features = text_embedding / text_embedding.norm(dim=-1, keepdim=True)
                    # similarity = (image_features @ text_features.T).squeeze(0)  # shape: (num_texts,)
                    # print(f"category {goal_val.object_category} similarity: {similarity}")
                    # t_debug += time.perf_counter() - t2
                    # # ----- for debug end

                    img_embedding =  img_embedding.squeeze().detach().cpu().numpy()
                    metadata = dict(
                        hfov=img_goal.hfov,
                        object_id=goal_val.object_id,
                        position=img_goal.position,
                        rotation=img_goal.rotation,
                        goal_id=goal_idx,
                        embedding=img_embedding,
                    )
                    goals_meta.append(metadata)


                # scene_id = goal_k.split("_")[0].split(".")[0]
                scene_id = goal_k.split("_")[0]
                object_id = goal_val.object_id
                data_goal[f"{scene_id}_{object_id}"] = goals_meta

        out_path = os.path.join(
            self.output_path, f"{scene}_{self.encoder_name}_embedding.pkl"
        )
        t_save_start = time.perf_counter()
        save_pickle(data_goal, out_path)
        t_save = time.perf_counter() - t_save_start

        t_total = time.perf_counter() - t_total_start
        t_other = t_total - (t_env + t_noise + t_encode + t_debug + t_save)
        ips = (n_images / t_encode) if t_encode > 0 else 0.0

        print(
            "[timing] "
            f"scene={scene} | total={t_total:.3f}s | env={t_env:.3f}s | "
            f"noise={t_noise:.3f}s | encode={t_encode:.3f}s "
            f"(imgs={n_images}, {ips:.2f} img/s) | "
            f"debug={t_debug:.3f}s | save={t_save:.3f}s | other={t_other:.3f}s"
        )

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
    
    def load_1(self, scene):
        scene = "2Pc8W48bu21"
        out_path_clip = os.path.join(
            self.output_path, f"{scene}_CLIP_iin_embedding.pkl"
        )
        data1 = load_pickle(out_path_clip)
        clip_image_count = 0
        for k ,items in data1.items():
            # print(f"clip: {k}, len {len(items)}")
            clip_image_count += len(items)
        print(clip_image_count)

        out_path = os.path.join(
            self.output_path, f"{scene}_{self.encoder_name}_embedding.pkl"
        )
        data = load_pickle(out_path)

        siglip_image_count = 0
        for k ,items in data.items():
            # print(f"siglip: {k}, len {len(items)}")
            siglip_image_count += len(items)
        print(siglip_image_count)

        train_dataset_json = f"data/datasets/ovon/hm3d/v2_new/train/content/{scene}.json.gz"
        dataset = load_dataset(train_dataset_json) 
        
        
        dataset_image_goal_count = 0
        goals = dataset["goals_by_category"]
        for k, category_goals in goals.items():
            for goal in category_goals:
                dataset_image_goal_count += len(goal["image_goals"])
                # print(f"dataset, object_id {goal['object_id']}, len {len(goal['image_goals'])}")
        print(dataset_image_goal_count)
        print(len(data))


    def run_all_scene_cache_image_goals(self):

        # 这里要同步去修改配置文件里面的路径
        # 获取split的值
        # split = "train"
        # split = "val_seen"
        # split = "val_seen_synonyms"
        split = "val_unseen"
        # 凭借split的路径
        data_path = "./data/datasets/ovon/hm3d/v2_new"
        # data_path = "./data/datasets/ovon/hm3d/v2_new"
        train_dataset_path = f"{data_path}/{split}/content"

        # 获取split的scene列表
        suffix = ".json.gz"
        scenes = sorted(
            p.name[:-len(suffix)]
            for p in Path(train_dataset_path).glob("*.json.gz")
            if p.is_file()
        )

        # 一步步调用run_cache_image_goals
        for scene in scenes:
            self.output_path = f"./tmp/iin/{split}_embeddings"
            os.makedirs(self.output_path, exist_ok=True)
            #self.run_cache_image_goals(scene=scene)
            self.run_cache_image_goals_with_concurrent(scene=scene, batch_size=256)
    

    # 并发 + 每个scene的image goal的统计
    def run_cache_image_goals_with_concurrent(self, scene, batch_size):
        t_total_start = time.perf_counter()

        out_path = os.path.join(
            self.output_path, f"{scene}_{self.encoder_name}_embedding.pkl"
        )
        if self.skip_if_exists and os.path.exists(out_path):
            print(f"[skip] {out_path} already exists")
            return out_path

        data_goal = {}
        t_env_start = time.perf_counter()
        with self._safe_env(scene) as env:
            t_env = time.perf_counter() - t_env_start

            goals = env._dataset.goals_by_category
            sensor = env.task.sensor_suite.sensors.get(
                "goat_subtask_raw_goal")
            if sensor is None:
                raise RuntimeError("Cannot find goal image sensor.")

            n_images = 0
            t_noise = 0.0
            t_encode = 0.0
            t_sensor = 0.0   # 你用于 text 相似度的调试耗时
            t_save = 0.0

            buf_imgs, buf_meta = [], []

            def flush():
                if not buf_imgs:
                    return
                emb = self.encoder.batch_encode_images(buf_imgs,batch_size=batch_size) 
                for e, m in zip(emb, buf_meta):
                    md = dict(
                        hfov=m["hfov"],
                        object_id=m["object_id"],
                        position=m["position"],
                        rotation=m["rotation"],
                        goal_id=m["goal_id"],
                        embedding=e,
                    )
                    # text = [f"a photo of f{m['object_category']}", "a photo of dog"]
                    # text_embedding = self.encoder.encode_text(text)
                    # t = torch.from_numpy(e)
                    # t = t.unsqueeze(0).to(self.device)
                    # t = t / t.norm(dim=-1, keepdim=True)
                    # text_features = text_embedding / text_embedding.norm(dim=-1, keepdim=True)
                    # similarity = (t @ text_features.T).squeeze(0)  # shape: (num_texts,)
                    # print(f"category {m['object_category']} similarity: {similarity}")
                
                    data_goal.setdefault(m["key"], []).append(md)
                buf_imgs.clear()
                buf_meta.clear()

            for goal_k, goal_vals in goals.items():
                scene_id = goal_k.split("_")[0]
                for goal_val in goal_vals:
                    if not getattr(goal_val, "image_goals", None):
                        continue

                    object_id = goal_val.object_id
                    if object_id == "pool table_414":
                        print('very intersting')

                    key = f"{scene_id}_{object_id}"

                    t_sensor_start = time.perf_counter()
                    n_images += len(goal_val.image_goals)
                    for goal_idx, img_goal in enumerate(goal_val.image_goals):
                        img = sensor._get_instance_image_goal(img_goal)

                        if self.add_noise:
                            img = self.apply_noise(img)

                        buf_imgs.append(img)
                        buf_meta.append(dict(
                            key=key,
                            hfov=img_goal.hfov,
                            object_id=object_id,
                            position=img_goal.position,
                            rotation=img_goal.rotation,
                            goal_id=goal_idx,
                            object_category=goal_val.object_category
                        ))

                        if len(buf_imgs) >= batch_size:
                            t_sensor += time.perf_counter() - t_sensor_start
                            t_encode_start = time.perf_counter()
                            flush()
                            t_encode += time.perf_counter() - t_encode_start
                            t_sensor_start = time.perf_counter()


            t_encode_start = time.perf_counter()
            flush()
            t_encode += time.perf_counter() - t_encode_start
        t_save_start = time.perf_counter() 
        save_pickle(data_goal, out_path)        
        t_save += time.perf_counter() - t_save_start

        t_total = time.perf_counter() - t_total_start
        t_other = t_total - (t_env + t_noise + t_encode + t_save + t_sensor)
        ips = (n_images / t_encode) if t_encode > 0 else 0.0

        msg = (
            "[timing] "
            f"scene={scene} | total={t_total:.3f}s | env={t_env:.3f}s | "
            f"noise={t_noise:.3f}s | encode={t_encode:.3f}s "
            f"(imgs={n_images}, {ips:.2f} img/s) | "
            f"save={t_save:.3f}s | other={t_other:.3f}s | "
            f"sensor={t_sensor:.3f}s | batch_size={batch_size}"
        )
        print(msg)
        with open("./tmp/monitor.txt", "a", encoding="utf-8") as f:
            f.write(msg + "\n")  # 文件追加写入


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
        default="./data/datasets/ovon/hm3d/v2_new",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="./tmp/iin/train_embeddings",
    )
    parser.add_argument(
        "--scene",
        type=str,
        # default="1S7LAXRdDqK",
        default="*",
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
    # cache.load_1(args.scene)
    cache.run_cache_object_goals()
    # cache.run_all_scene_cache_image_goals()
    # cache.run_cache_image_goals_with_concurrent("2Pc8W48bu21", 256)
