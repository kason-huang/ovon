import habitat_sim

# 创建仿真器配置
cfg = habitat_sim.SimulatorConfiguration()
cfg.scene_dataset_config_file = 'data/scene_datasets/hm3d/hm3d_annotated_basis.scene_dataset_config.json'
cfg.scene_id = '00016-qk9eeNeR4vw'
cfg.gpu_device_id = 0

# 创建 agent 配置（必需）
agent_cfg = habitat_sim.AgentConfiguration()

# 使用 Configuration 包装配置并创建仿真器
sim = habitat_sim.Simulator(habitat_sim.Configuration(cfg, [agent_cfg]))

print(f'物体数量: {len(sim.semantic_scene.objects)}')

# 关闭仿真器
sim.close()