from legged_gym import LEGGED_GYM_ROOT_DIR, envs
from .legged_robot import LeggedRobot
import torch
import numpy as np

import os
from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgym.torch_utils import *


class LeggedRobot_reach(LeggedRobot):
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

        self._tar_speed = self.cfg.env.tarSpeed
        self._tar_change_steps_min = self.cfg.env.tarChangeStepsMin
        self._tar_change_steps_max = self.cfg.env.tarChangeStepsMax
        
        self._tar_dist_max = self.cfg.env.tarDistMax
        self._tar_height_min = self.cfg.env.tarHeightMin
        self._tar_height_max = self.cfg.env.tarHeightMax

        self._tar_change_steps = torch.zeros([self.num_envs], device=self.device, dtype=torch.int64)
        self._tar_pos = torch.zeros([self.num_envs, 3], device=self.device, dtype=torch.float)

        reach_body_name = "right_rubber_hand"
        #right_foot_name = 
        self._reach_body_id = self._build_reach_body_id_tensor(self.envs[0], self.actor_handles[0], reach_body_name)

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

        self.marker_pos[..., 0:3] = self._tar_pos
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(self.marker_actor_ids), len(self.marker_actor_ids))
        return
    
    def _update_task(self):
        reset_task_mask = self.episode_length_buf >= self._tar_change_steps
        rest_env_ids = reset_task_mask.nonzero(as_tuple=False).flatten()
        if len(rest_env_ids) > 0:
            self._reset_task(rest_env_ids)
        return


    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)

        if len(env_ids)>0:
            self._reset_task(env_ids)
            self._update_marker()
        return

    def _build_reach_body_id_tensor(self, env_ptr, actor_handle, body_name):
        body_id = self.gym.find_actor_rigid_body_handle(env_ptr, actor_handle, body_name)
        assert(body_id != -1)
        body_id = to_torch(body_id, device=self.device, dtype=torch.long)
        return body_id
    
    def _resample_commands(self, env_ids): #commands 一直是0
        return
    
    def _reset_task(self, env_ids):
        n = len(env_ids)

        a = torch.rand(n, device=self.device, dtype=torch.float32) * 0.5
        b = torch.rand(n, device=self.device, dtype=torch.float32)* 0.75 - 0.5
        x_noises = a * torch.cos(np.pi * b)
        y_noises = a * torch.sin(np.pi * b)
        rand_pos = torch.cat([x_noises.unsqueeze(1), y_noises.unsqueeze(1), torch.rand((n, 1), device=self.device, dtype=torch.float32)],dim=1)
        rand_pos[..., 2] = (self._tar_height_max - self._tar_height_min) * rand_pos[..., 2] + self._tar_height_min
        rand_pos = quat_rotate(self.robot_root_states[env_ids, 3:7], rand_pos)
        rand_pos[:, :2] += self.robot_root_states[env_ids, :2]
        
        change_steps = torch.randint(low=self._tar_change_steps_min, high=self._tar_change_steps_max,
                                     size=(n,), device=self.device, dtype=torch.int64)
        self._tar_pos[env_ids, :] = rand_pos
        self._tar_change_steps[env_ids] = self.episode_length_buf[env_ids] + change_steps
        return
    
    def _reward_reach(self):
        # type: (Tensor, Tensor, Tensor, float, float) -> Tensor
        pos_err_scale = 4.0
        reach_body_pos = self.rigid_body_states[:, self._reach_body_id, :3]

        pos_diff = self._tar_pos - reach_body_pos
        pos_err = torch.sum(pos_diff * pos_diff, dim=-1)
        pos_reward = torch.exp(-pos_err_scale * pos_err)
        
        reward = pos_reward

        return reward
    
    def compute_observations(self):
        """ Computes observations
        """
        local_tar_pos = compute_location_observations(self.robot_root_states, self._tar_pos)
        current_obs = torch.cat(( 
                                    self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity, #四元数
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos, #dof_pos
                                    self.dof_vel * self.obs_scales.dof_vel, #dof_vel
                                    self.actions, #dof_target
                                    local_tar_pos
                                    ),dim=-1)
        if current_obs.isnan().any():
            print("has nan!")
            current_obs = torch.zeros((self.envs, 6 + self.num_dof * 2 + self.num_actions  + 3), dtype = torch.float, devices=self.device)

        # add noise if needed
        current_actor_obs = torch.clone(current_obs)
        if self.add_noise:
            current_actor_obs = current_actor_obs + (2 * torch.rand_like(current_actor_obs) - 1) * self.noise_scale_vec[0:(6 + 2 * self.num_dof + self.num_actions + 3)]

        self.obs_buf = torch.cat((self.obs_buf[:, self.num_one_step_obs:self.actor_obs_length], current_actor_obs), dim=-1) # 一共六次的
        
        self.privileged_obs_buf = current_obs

    def compute_termination_observations(self, env_ids):
        """ Computes observations
        """
        local_tar_pos = compute_location_observations(self.robot_root_states, self._tar_pos)
        current_obs = torch.cat(( 
                                    self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions,
                                    # self.negative_buf.unsqueeze(-1),
                                    local_tar_pos
                                    ),dim=-1)
        return current_obs[env_ids]


    def step(self, actions):
        self._update_task()
        return super().step(actions)
    
    def post_physics_step(self):
        env_ids, termination_privileged_obs = super().post_physics_step()
        self.left_feet_vel = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[:, self.left_feet_indices, 7:10]
        self.right_feet_vel = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[:, self.right_feet_indices, 7:10]
        self.left_feet_vel_norm = torch.norm(self.left_feet_vel, dim=-1)
        self.right_feet_vel_norm = torch.norm(self.right_feet_vel, dim=-1)

        return env_ids, termination_privileged_obs
    
    def _init_buffers(self):
        super()._init_buffers()
        self.left_feet_vel = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[:, self.left_feet_indices, 7:10]
        self.right_feet_vel = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[:, self.right_feet_indices, 7:10]
        self.left_feet_vel_norm = torch.norm(self.left_feet_vel, dim=-1)
        self.right_feet_vel_norm = torch.norm(self.right_feet_vel, dim=-1)
    
    def _reward_constraint_foot_vel(self):
        
        left_foot_error = torch.sum(self.left_feet_vel_norm, dim=1)
        right_foot_error = torch.sum(self.right_feet_vel_norm, dim=1)
        return torch.exp(-left_foot_error/self.cfg.rewards.tracking_sigma) + torch.exp(-right_foot_error/self.cfg.rewards.tracking_sigma)

    

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
def calc_heading_quat_inv(q):
    # type: (Tensor) -> Tensor
    # calculate heading rotation from quaternion
    # the heading is the direction on the xy plane
    # q must be normalized
    heading = calc_heading(q)
    axis = torch.zeros_like(q[..., 0:3])
    axis[..., 2] = 1

    heading_q = quat_from_angle_axis(-heading, axis)
    return heading_q

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

@torch.jit.script
def compute_location_observations(root_states, tar_pos):
    # type: (Tensor, Tensor) -> Tensor
    root_rot = root_states[:, 3:7]
    root_pos = root_states[:, 0:3]
    heading_rot = calc_heading_quat_inv(root_rot)
    local_tar_pos = quat_rotate(heading_rot, tar_pos - root_pos)

    obs = local_tar_pos
    return obs
