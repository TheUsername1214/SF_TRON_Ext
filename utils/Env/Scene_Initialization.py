import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, ContactSensorCfg, RayCasterCfg, ImuCfg, patterns, RayCasterCameraCfg
from isaaclab.utils import configclass
from isaaclab.terrains import TerrainImporter, TerrainImporterCfg, TerrainGeneratorCfg
from isaaclab.terrains.height_field.hf_terrains_cfg import HfSteppingStonesTerrainCfg, HfRandomUniformTerrainCfg
from isaaclab.sim import DomeLightCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
import numpy as np


def env_setup(file_path, dt, sub_step, agents_num, device):
    """初始化Isaac Sim 世界"""
    sub_terrains = {}
    iterations = 5
    heights = np.linspace(0, 0.15, iterations)  # 给太多等级不好，生产地形基本不生成高地
    distance = np.linspace(0.1, 0.2, iterations)  # 给太多等级不好，生产地形基本不生成高地
    for i in range(len(heights)):
        for j in range(len(distance)):
            # sub_terrains[f"stepping_stone_{i + 1}_{j + 1}"] = HfSteppingStonesTerrainCfg(
            #     proportion=1.0,
            #     border_width=0.1,
            #     holes_depth=-0.3,  # 这个不能给高，不然碰撞计算要花很久
            #     stone_height_max=0,
            #     stone_width_range=(0.5, 0.8),
            #     stone_distance_range=(0.1, distance[j]),
            #     platform_width=0.8,
            # )
            sub_terrains[f"flat_plane_{i + 1}_{j + 1}"] = HfRandomUniformTerrainCfg(
                proportion=1.0,
                border_width=0.1,
                noise_range=(-heights[i] / 10, heights[i] / 10),
                noise_step=0.01,
            )

    num_row = 20

    print(f"Environment initialization: num_row of terrains: {num_row}x{num_row}")

    gen_cfg = TerrainGeneratorCfg(
        num_rows=num_row,  # 太多了会出问题，比如碰撞体失效
        num_cols=num_row,
        size=(10, 3),
        color_scheme="none",
        sub_terrains=sub_terrains,
        curriculum=False,
        border_width=10,
    )

    @configclass
    class SensorsSceneCfg(InteractiveSceneCfg):
        """Design the scene with sensors on the robot."""

        """环境中的光照设置"""

        light = AssetBaseCfg(
            prim_path="/World/skyLight",
            spawn=DomeLightCfg(
                intensity=750.0,
                color=(0.9, 0.9, 0.9),
                texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
            ),
        )
        """地形设置"""
        imp_cfg = TerrainImporterCfg(
            prim_path="/World/defaultGroundPlane",
            terrain_type="generator",
            terrain_generator=gen_cfg,
            debug_vis=False,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1,
                dynamic_friction=1
            )
        )

        """机器人设置"""
        robot = ArticulationCfg(
            spawn=sim_utils.UsdFileCfg(
                usd_path=file_path,
                activate_contact_sensors=1),
            prim_path="{ENV_REGEX_NS}/Robot",
            actuators={"wheel_acts": ImplicitActuatorCfg(joint_names_expr=[".*"], damping=None, stiffness=None)},
            collision_group=0,

        )

        # sensors
        """高度扫描仪"""
        height_scanner = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_Link",
            update_period=0.1,
            offset=RayCasterCfg.OffsetCfg(pos=(0.3, 0.0, 20)),
            ray_alignment="yaw",
            pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1]),
            debug_vis=True,
            mesh_prim_paths=["/World/defaultGroundPlane"]
        )

        """脚部传感器"""
        L_contact_sensor = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/ankle_L_Link",
            update_period=0.0, history_length=1, debug_vis=False
        )

        R_contact_sensor = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/ankle_R_Link",
            update_period=0.0, history_length=1, debug_vis=False
        )

        imu_sensor = ImuCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_Link",
            update_period=0,
        )

        L_imu_sensor = ImuCfg(
            prim_path="{ENV_REGEX_NS}/Robot/ankle_L_Link",
            update_period=0,
        )

        R_imu_sensor = ImuCfg(
            prim_path="{ENV_REGEX_NS}/Robot/ankle_R_Link",
            update_period=0,
        )

        # Depth_Camera = RayCasterCameraCfg(
        #     prim_path="{ENV_REGEX_NS}/Robot/Camera_Frame",
        #     update_period=0,
        #     mesh_prim_paths=["/World/defaultGroundPlane"],
        #     max_distance=6,
        #     depth_clipping_behavior="max",
        #     debug_vis=True,
        #     offset = RayCasterCameraCfg.OffsetCfg(convention = "world"),
        #     pattern_cfg=patterns.PinholeCameraPatternCfg(width=11,
        #                                                  height=16)
        # )

    """初始化Isaac Sim 世界"""
    # 启动Isaac Sim 软件， 必须放在导入Isaac sim 库之前。
    sim_cfg = sim_utils.SimulationCfg(dt=dt / sub_step, device=device)
    sim = sim_utils.SimulationContext(sim_cfg)
    # design scene
    scene_cfg = SensorsSceneCfg(num_envs=agents_num, env_spacing=0)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    scene.reset()
    scene.update(dt=0)

    return sim, scene
