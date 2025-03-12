# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from .base_config import BaseConfig

class LeggedRobotCfg(BaseConfig):
    class env:
        blind = False
        num_envs = 6177

        num_actor_history = 6
        

        num_one_step_perception = 104

        num_privileged_perception = 187

        if blind:
            num_one_step_perception = 0

        num_one_step_observations = 66 + num_one_step_perception
        num_privileged_obs = 66 + 3 + num_privileged_perception

        num_observations = num_actor_history * num_one_step_observations



        num_actions = 19 # number of actuators on robot
        action_curriculum = True
        env_spacing = 3.  # not used with heightfields/trimeshes 
        send_timeouts = True # send time out information to the algorithm
        episode_length_s = 20 # episode length in seconds

        next_goal_threshold = 0.2
        reach_goal_delay = 0.1
        num_future_goal_obs = 2

    class terrain:
        mesh_type = 'trimesh' # "heightfield" # none, plane, heightfield or trimesh
        hf2mesh_method = "grid"  # grid or fast
        max_error = 0.1 # for fast
        max_error_camera = 2

        y_range = [-0.4, 0.4]
        
        edge_width_thresh = 0.05
        horizontal_scale = 0.05 # [m] influence computation time by a lot
        horizontal_scale_camera = 0.1
        vertical_scale = 0.005 # [m]
        border_size = 5 # [m]
        height = [0.02, 0.06]
        simplify_grid = False
        gap_size = [0.02, 0.1]
        stepping_stone_distance = [0.02, 0.08]
        downsampled_scale = 0.075
        curriculum = True

        all_vertical = False
        no_flat = True
        
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.
        measure_heights = True
        measure_interval = 5
        height_buffer_len = 4
        measured_points_x = [-0.35, -0.25, -0.15, -0.05, 0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85] # [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2] 
        measured_points_y = [-0.35, -0.25, -0.15, -0.05, 0.05, 0.15, 0.25, 0.35] # [-0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4]
        critic_measured_points_x = [-0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1]
        critic_measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

        measure_horizontal_noise = 0.0

        selected = False # select a unique terrain type and pass all arguments
        terrain_kwargs = None # Dict of arguments for selected terrain
        max_init_terrain_level = 5 # starting curriculum state
        terrain_length = 18.
        terrain_width = 4
        num_rows= 10 # number of terrain rows (levels)  # spreaded is benifitiall !
        num_cols =40 # number of terrain cols (types)
        
        terrain_dict = {"parkour_flat": 0.2,
                        "parkour_hurdle": 0.2,
                        "parkour_wall": 0.2,
                        "parkour_step": 0.2,
                        "parkour_gap": 0.2}


        terrain_proportions = list(terrain_dict.values())
        
        # trimesh only:
        slope_treshold = 1.5# slopes above this threshold will be corrected to vertical surfaces
        origin_zero_z = True

        num_goals = 8

    class commands:
        curriculum = True
        max_curriculum = 2.0
        num_commands = 1 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10. # time before command are changed[s]
        class ranges:
            lin_vel = [-0.0, 0.0] # min max [m/s]

        waypoint_delta = 0.7
    class init_state:
        pos = [0.0, 0.0, 1.] # x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0] # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]
        default_joint_angles = { # target angles when action = 0.0
            "joint_a": 0., 
            "joint_b": 0.}

    class control:
        control_type = 'P' # P: position, V: velocity, T: torques
        # PD Drive parameters:
        stiffness = {'joint_a': 10.0, 'joint_b': 15.}  # [N*m/rad]
        damping = {'joint_a': 1.0, 'joint_b': 1.5}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.5
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4
        hip_reduction = 1.0
        curriculum_joints = []

    class asset:
        file = ""
        name = "legged_robot"  # actor name
        foot_name = "None" # name of the feet bodies, used to index body state and contact force tensors
        penalize_contacts_on = []
        terminate_after_contacts_on = []
        disable_gravity = False
        collapse_fixed_joints = True # merge bodies connected by fixed joints. Specific fixed joints can be kept by adding " <... dont_collapse="true">
        fix_base_link = False # fixe the base of the robot
        default_dof_drive_mode = 3 # see GymDofDriveModeFlags (0 is none, 1 is pos tgt, 2 is vel tgt, 3 effort)
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        replace_cylinder_with_capsule = True # replace collision cylinders with capsules, leads to faster/more stable simulation
        flip_visual_attachments = True # Some .obj meshes must be flipped from y-up to z-up
        
        density = 0.001
        angular_damping = 0.
        linear_damping = 0.
        max_angular_velocity = 1000.
        max_linear_velocity = 1000.
        armature = 0.
        thickness = 0.01

    class domain_rand:
        
        randomize_joint_injection = False
        joint_injection_range = [-0.1, 0.1]
        
        randomize_actuation_offset = False
        actuation_offset_range = [-0.1, 0.1]

        randomize_payload_mass = False
        payload_mass_range = [-5, 10]

        randomize_com_displacement = False
        com_displacement_range = [-0.1, 0.1]

        randomize_link_mass = False
        link_mass_range = [0.7, 1.3]
        
        randomize_friction = False
        friction_range = [0.1, 1.25]
        
        randomize_restitution = False
        restitution_range = [0.1, 1.0]
        
        randomize_kp = False
        kp_range = [0.8, 1.2]
        
        randomize_kd = False
        kd_range = [0.8, 1.2]
        
        randomize_initial_joint_pos = False
        initial_joint_pos_scale = [0.5, 1.5]
        initial_joint_pos_offset = [-0.1, 0.1]
        
        push_robots = False
        push_interval_s = 5
        max_push_vel_xy = 1.

        delay = False

    class rewards:

        class scales:
            # tracking rewards
            tracking_goal_vel = 5.0
            tracking_yaw = 5.0
            # regularization rewards
            lin_vel_z = -1.0
            ang_vel_xy = -0.05
            orientation = -2.
            dof_acc = -2.5e-7
            collision = -12.
            action_rate = -0.3
            delta_torques = -1.0e-7
            feet_stumble = -1
            feet_edge = -1.5
            torques = -1e-5 # -1e-4
            dof_vel = -5e-4
            feet_air_time = 4.0

            dof_pos_limits = -2.0 #-2.0
            dof_vel_limits = -1.0 #0.0
            torque_limits = -1.0 #0.0

            deviation_all_joint = -0.0
            deviation_arm_joint = -0.5
            deviation_leg_joint = -0.0
            deviation_hip_joint = -5.0
            deviation_waist_joint = -5.0
            deviation_ankle_joint = -0.0
            feet_distance = 1.0
            feet_distance_lateral = 10.0
            knee_distance_lateral = 5.0
            feet_clearance = -1.0
            feet_ground_parallel = -10.0


        only_positive_rewards = False # if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)
        soft_dof_pos_limit = 0.9 # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 0.9
        soft_torque_limit = 0.9
        base_height_target = 1.0
        max_contact_force = 500. # forces above this value are penalized
        clearance_height_target = 0.25
        least_feet_distance = 0.30
        least_feet_distance_lateral = 0.30
        least_knee_distance_lateral = 0.30

    class normalization:
        class obs_scales:
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0
        clip_observations = 100.
        clip_actions = 100.

    class noise:
        add_noise = True
        noise_level = 1.0 # scales other values
        class noise_scales:
            dof_pos = 0.01
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05
            height_measurements = 0.1

    # viewer camera:
    class viewer:
        ref_env = 0
        pos = [10, 0, 6]  # [m]
        lookat = [11., 5, 3.]  # [m]

    class sim:
        dt =  0.005
        substeps = 1
        gravity = [0., 0. ,-9.81]  # [m/s^2]
        up_axis = 1  # 0 is y, 1 is z

        class physx:
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 8
            num_velocity_iterations = 0
            contact_offset = 0.01  # [m]
            rest_offset = 0.0   # [m]
            bounce_threshold_velocity = 0.5 #0.5 [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23 #2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            contact_collection = 2 # 0: never, 1: last sub-step, 2: all sub-steps (default=2)

class LeggedRobotCfgPPO(BaseConfig):
    seed = 1
    runner_class_name = 'HIMOnPolicyRunner'
    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 256]
        critic_hidden_dims = [512, 256, 256]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        # only for 'ActorCriticRecurrent':
        # rnn_type = 'lstm'
        # rnn_hidden_size = 512
        # rnn_num_layers = 1
        
    class algorithm:
        # training params
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4 # mini batch size = num_envs*nsteps / nminibatches
        learning_rate = 1.e-3 #5.e-4
        schedule = 'adaptive' # could be adaptive, fixed
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.

    class runner:
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'HIMPPO'
        num_steps_per_env = 100 # per iteration
        max_iterations = 200000 # number of policy updates

        # logging
        save_interval = 100 # check for potential saves every this many iterations
        experiment_name = 'test'
        run_name = ''
        # load and resume
        resume = False
        load_run = -1 # -1 = last run
        checkpoint = -1 # -1 = last saved model
        resume_path = None # updated from load_run and chkpt