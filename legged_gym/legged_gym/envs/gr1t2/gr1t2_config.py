from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class GR1T2Cfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_envs = 4096
        num_one_step_observations = 78
        num_one_step_privileged_obs = 78 + 3
        num_actor_history = 6
        num_critic_history = 1
        num_observations = num_actor_history * num_one_step_observations + 81
        num_privileged_obs = num_critic_history * num_one_step_privileged_obs + 187
        num_actions = 23 # number of actuators on robot
        action_curriculum = True
        env_spacing = 3.  # not used with heightfields/trimeshes 
        send_timeouts = True # send time out information to the algorithm
        episode_length_s = 20 # episode length in seconds
        
    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'trimesh' # "heightfield" # none, plane, heightfield or trimesh
        edge_width_thresh = 0.05
        horizontal_scale = 0.1 # [m]
        vertical_scale = 0.005 # [m]
        border_size = 25 # [m]
        curriculum = True
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.
        # rough terrain only:
        measure_heights = True
        measure_interval = 5
        height_buffer_len = 4
        measured_points_x = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2] 
        measured_points_y = [-0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4]
        critic_measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        critic_measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        selected = False # select a unique terrain type and pass all arguments
        terrain_kwargs = None # Dict of arguments for selected terrain
        max_init_terrain_level = 5 # starting curriculum state
        terrain_length = 8.
        terrain_width = 8.
        num_rows= 10 # number of terrain rows (levels)
        num_cols = 10 # number of terrain cols (types)
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
        terrain_proportions = [0.2, 0.8]
        # terrain_proportions = [0.1, 0.2, 0.3, 0.3, 0.1]
        # terrain_proportions = [0.1, 0.2, 0.2, 0.2, 0.15, 0.15]
        # trimesh only:
        slope_treshold = 0.75 # slopes above this threshold will be corrected to vertical surfaces

    class commands(LeggedRobotCfg.commands):
        curriculum = False
        max_curriculum = 2.5
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10. # time before command are changed[s]
        heading_command = True # if true: sample heading command, if false: sample ang_vel_yaw
        heading_to_ang_vel = True # if true: compute ang vel command from heading error
        class ranges:
            lin_vel_x = [-1.0, 1.0] # min max [m/s]
            lin_vel_y = [-0.0, 0.0] # min max [m/s]
            ang_vel_yaw = [-1.0, 1.0] # min max [rad/s]
            heading = [-3.14, 3.14] # min max [rad]


    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 1.0] # x,y,z [m]
        default_joint_angles = {  # = target angles [rad] when action = 0.0
            # left leg
            'l_hip_roll': 0.0,
            'l_hip_yaw': 0.0,
            'l_hip_pitch': -0.5236,
            'l_knee_pitch': 1.0472,
            'l_ankle_pitch': -0.5236,
            'l_ankle_roll': 0.0,

            # right leg
            'r_hip_roll': 0.0,
            'r_hip_yaw': 0.0,
            'r_hip_pitch': -0.5236,
            'r_knee_pitch': 1.0472,
            'r_ankle_pitch': -0.5236,
            'r_ankle_roll': 0.0,

            # waist
            'joint_waist_yaw': 0.0,
            'joint_waist_pitch': 0.0,
            'joint_waist_roll': 0.0,

            # left arm
            'l_shoulder_pitch': 0.0,
            'l_shoulder_roll': 0.0,
            'l_shoulder_yaw': 0.0,
            'l_elbow_pitch': -0.3,

            # right arm
            'r_shoulder_pitch': 0.0,
            'r_shoulder_roll': 0.0,
            'r_shoulder_yaw': 0.0,
            'r_elbow_pitch': -0.3,
        }

    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        control_type = 'P'
          # PD Drive parameters:
        stiffness = {
            'hip_roll': 251.625, 'hip_yaw': 362.5214, 'hip_pitch': 200,
            'knee_pitch': 200,
            'ankle_pitch': 10.9805, 'ankle_roll': 20,
            'waist_yaw': 362.5214, 'waist_pitch': 362.5214, 'waist_roll':362.5214,

            'shoulder_pitch': 92.85, 'shoulder_roll': 92.85, 'shoulder_yaw': 112.06,
            'elbow_pitch': 112.06
        }  # [N*m/rad]
        damping = {
            'hip_roll': 14.72, 'hip_yaw': 10.0833, 'hip_pitch': 11,
            'knee_pitch': 11,
            'ankle_pitch': 0.5991, 'ankle_roll': 1.0,
            'waist_yaw': 10.0833, 'waist_pitch': 10.0833, 'waist_roll': 10.0833,
            'shoulder_pitch': 2.575, 'shoulder_roll': 2.575, 'shoulder_yaw': 3.1,
            'elbow_pitch': 3.1
        } # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4
        curriculum_joints = ['joint_waist_pitch', 'joint_waist_roll', 'joint_waist_yaw', 'l_shoulder_roll', 'r_shoulder_roll', 'l_shoulder_yaw', 'r_shoulder_yaw']
        left_leg_joints = ['l_hip_roll', 'l_hip_yaw', 'l_hip_pitch', 'l_knee_pitch', 'l_ankle_pitch', 'l_ankle_roll']
        right_leg_joints = ['r_hip_roll', 'r_hip_yaw', 'r_hip_pitch', 'r_knee_pitch', 'r_ankle_pitch', 'r_ankle_roll']
        left_arm_joints = ['l_shoulder_pitch', 'l_shoulder_roll', 'l_shoulder_yaw', 'l_elbow_pitch']
        right_arm_joints = ['r_shoulder_pitch', 'r_shoulder_roll', 'r_shoulder_yaw', 'r_elbow_pitch']
        upper_body_link = "base"

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/gr1t2/urdf/gr1t2.urdf'
        name = "gr1t2"
        foot_name = "foot_roll"
        left_foot_name = "left_foot"
        right_foot_name = "right_foot"
        penalize_contacts_on = ["hip", "knee", "torso", "shoulder", "elbow", "pelvis"]
        terminate_after_contacts_on = []
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False
        
    class domain_rand(LeggedRobotCfg.domain_rand):
        
        randomize_joint_injection = True
        joint_injection_range = [-0.01, 0.01]
        
        randomize_actuation_offset = True
        actuation_offset_range = [-0.01, 0.01]

        randomize_payload_mass = True
        payload_mass_range = [-5, 10]

        randomize_com_displacement = True
        com_displacement_range = [-0.1, 0.1]

        randomize_link_mass = True
        link_mass_range = [0.8, 1.2]
        
        randomize_friction = True
        friction_range = [0.1, 2.0]
        
        randomize_restitution = True
        restitution_range = [0.0, 1.0]
        
        randomize_kp = True
        kp_range = [0.8, 1.2]
        
        randomize_kd = True
        kd_range = [0.8, 1.2]
        
        randomize_initial_joint_pos = True
        initial_joint_pos_scale = [0.5, 1.5]
        initial_joint_pos_offset = [-0.1, 0.1]
        
        push_robots = True
        push_interval_s = 5
        max_push_vel_xy = 1.5

        delay = True
        
    class rewards( LeggedRobotCfg.rewards ):
        class scales:
            termination = -0.0
            tracking_lin_vel = 4.0
            tracking_ang_vel = 4.0
            tracking_yaw = 0.0
            lin_vel_z = -2.0
            ang_vel_xy = -0.1
            orientation = -2.0 # -2.0
            dof_acc = -2.5e-7 # -2.5e-7
            joint_power = -1e-4
            base_height = -0.0
            base_height_wrt_feet = -2.0
            feet_clearance = -1.0
            action_rate = -0.01 # -0.02
            smoothness = -0.01 # -0.02
            feet_air_time = 0.0
            collision = -10.0
            feet_stumble = -0.5
            stand_still = -0.0
            torques = -1e-4 # -1e-4
            dof_vel = -1e-4 # -1e-4
            joint_tracking_error = -1.0
            joint_deviation = -0.1 #-1.0
            dof_pos_limits = -2.0 #-2.0
            dof_vel_limits = -0.0 #0.0
            torque_limits = -0.0 #0.0
            no_fly = 1.0 # 1.0
            leg_power_symmetry = -1e-3 # -1e-3
            arm_power_symmetry = -1e-3 # -1e-2
            feet_distance = 2.0
            feet_distance_lateral = 2.0
            feet_slip = -1.0 # -0.5
            feet_ground_parallel = -5.0
            feet_contact_forces = -1e-3 # -1e-3
            contact_momentum = -1e-3 # -1e-3

        only_positive_rewards = False # if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)
        soft_dof_pos_limit = 0.95 # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 0.85
        soft_torque_limit = 0.85
        base_height_target = 1.0
        max_contact_force = 500. # forces above this value are penalized
        clearance_height_target = 0.15
        least_feet_distance = 0.15
        least_feet_distance_lateral = 0.10

class GR1T2CfgPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
    class runner( LeggedRobotCfgPPO.runner ):
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'HIMPPO'
        num_steps_per_env = 100 # per iteration
        max_iterations = 200000 # number of policy updates

        # logging
        save_interval = 20 # check for potential saves every this many iterations
        run_name = 'HIM'
        experiment_name = 'gr1t2'
        wandb_project = "Humanoid Control"
        logger = 'tensorboard' # 'wandb' or 'tensorboard
        
        # load and resume
        resume = False
        load_run = -1 # -1 = last run
        checkpoint = -1 # -1 = last saved model
        resume_path = None # updated from load_run and chkpt
