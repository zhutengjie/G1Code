from legged_gym import LEGGED_GYM_ROOT_DIR, envs
from .legged_robot import LeggedRobot
import torch

import os
from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgym.torch_utils import *


class LeggedRobot_reach(LeggedRobot):
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

        if (not self.headless):
            self._build_marker_state_tensors()


    def _load_marker_asset(self):
        asset_path = self.cfg.asset.marker.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.angular_damping = 0.01
        asset_options.linear_damping = 0.01
        asset_options.max_angular_velocity = 100.0
        asset_options.density = 1.0
        asset_options.fix_base_link = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE

        self._marker_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

        return
    
    def _build_marker_state_tensors(self):
        num_actors = self.root_states.shape[0] // self.num_envs
        self.marker_states = self.root_states.view(self.num_envs, num_actors, self.root_states.shape[-1])[..., 1, :]
        self.marker_pos = self.marker_states[..., :3]
        
        self.marker_actor_ids = self.robot_actor_ids + 1

        return
    
    def _build_env(self, env_id, env_handle, robot_asset, rigid_shape_props_asset, dof_props_asset, start_pose):
        super()._build_env(env_id, env_handle, robot_asset, rigid_shape_props_asset, dof_props_asset, start_pose)

        if (not self.headless):
            self._build_marker(env_id, env_handle)
        return
    
    def _build_marker(self, env_id, env_ptr):
        col_group = env_id
        col_filter = 2
        segmentation_id = 0
        default_pose = gymapi.Transform()
        
        marker_handle = self.gym.create_actor(env_ptr, self._marker_asset, default_pose, "marker", col_group, col_filter, segmentation_id)
        self.gym.set_rigid_body_color(env_ptr, marker_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.8, 0.0, 0.0))
        self._marker_handles.append(marker_handle)

        return
    def _create_envs(self):
        if (not self.headless):
            self._marker_handles = []
            self._load_marker_asset()
        super()._create_envs()

        return
    
    def _update_marker(self):

        root_pos = self.robot_root_states[:, :3]

        tensor = torch.rand(self.num_envs, 3, dtype=torch.float32, device=self.device)  # 生成 [0,1) 的随机数
        tensor = tensor * torch.tensor([4, 4, 0.6],dtype=torch.float32, device=self.device) - torch.tensor([2, 2, 0.3],dtype=torch.float32, device=self.device)

        self.marker_pos[..., 0:3] = tensor + root_pos

        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(self.marker_actor_ids), len(self.marker_actor_ids))
        return
    
    def _update_task(self):
        pass

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)

        if len(env_ids)>0:
            self._update_marker()
        return 
    

    

    