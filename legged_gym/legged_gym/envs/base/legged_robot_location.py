from legged_gym import LEGGED_GYM_ROOT_DIR, envs
from .legged_robot import LeggedRobot
import torch
from utils import torch_utils

import os
from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgym.torch_utils import *


class LeggedRobot_reach(LeggedRobot):
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

        self.tar_speed = self.cfg.env.tarSpeed
        self._tar_dist_max = self.cfg.env.tarDistMax

        self._prev_root_pos = torch.zeros([self.num_envs, 3], device=self.device, dtype=torch.float)
        self._tar_pos = torch.zeros([self.num_envs, 2], device=self.device, dtype=torch.float)

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
    
    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)

        if len(env_ids)>0:
            self._reset_task(env_ids)
            if not self.headless:
                self._update_marker()
        return
    
    def _reset_task(self, env_ids):
        n = len(env_ids)

        char_root_pos = self.robot_root_states[env_ids, 0:2]
        rand_pos = self._tar_dist_max * (2.0 * torch.rand([n, 2], device=self.device) - 1.0)

        self._tar_pos[env_ids] = char_root_pos + rand_pos
        return
    
    def _reward_location(self):
        
        pos_err_scale = 0.5
        vel_err_scale = 4.0
        dist_threshold = 0.5

        pos_reward_w = 0.5
        vel_reward_w = 0.4
        face_reward_w = 0.1

        root_pos = self.robot_root_states[:, :2]
        pos_diff = self._tar_pos - root_pos
        pos_err = torch.sum(pos_diff * pos_diff, dim=-1)
        pos_reward = torch.exp(-pos_err_scale * pos_err)

        tar_dir = self._tar_pos - self.robot_root_states[:, :2]
        tar_dir = torch.nn.functional.normalize(tar_dir, dim=-1)
        vel = self.robot_root_states[:, 7:10]
        tar_dir_speed = torch.sum(tar_dir * vel[..., :2], dim=-1)
        tar_vel_err = self.tar_speed - tar_dir_speed
        tar_vel_err = torch.clamp_min(tar_vel_err, 0.0)
        vel_reward = torch.exp(-vel_err_scale * (tar_vel_err * tar_vel_err))
        speed_mask = tar_dir_speed <= 0
        vel_reward[speed_mask] = 0

        root_rot = self.robot_root_states[:, 3:7]
        heading_rot = calc_heading_quat(root_rot)
        facing_dir = torch.zeros_like(root_pos)
        facing_dir[..., 0] = 1.0
        facing_dir = quat_rotate(heading_rot, facing_dir)
        facing_err = torch.sum(tar_dir * facing_dir[..., 0:2], dim=-1)
        facing_reward = torch.clamp_min(facing_err, 0.0)

        dist_mask = pos_err < dist_threshold
        vel_reward[dist_mask] = 1.0
        vel_reward[dist_mask] = 1.0

        return pos_reward_w * pos_reward + vel_reward_w * vel_reward + face_reward_w * facing_reward
    
    def compute_observations(self):
        return super().compute_observations()

@torch.jit.script
def calc_heading(q):
    # type: (Tensor) -> Tensor
    # calculate heading direction from quaternion
    # the heading is the direction on the xy plane
    # q must be normalized
    ref_dir = torch.zeros_like(q[..., 0:3])
    ref_dir[..., 0] = 1
    rot_dir = quat_rotate(q, ref_dir)

    heading = torch.atan2(rot_dir[..., 1], rot_dir[..., 0])
    return heading

@torch.jit.script
def calc_heading_quat(q):
    # type: (Tensor) -> Tensor
    # calculate heading rotation from quaternion
    # the heading is the direction on the xy plane
    # q must be normalized
    heading = calc_heading(q)
    axis = torch.zeros_like(q[..., 0:3])
    axis[..., 2] = 1

    heading_q = quat_from_angle_axis(heading, axis)
    return heading_q