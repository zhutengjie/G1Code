from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class G1Cfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        blind = True
        num_envs = 4096

        num_actor_history = 6
        
        num_actions = 23 # number of actuators on robot
        num_dofs = 23
        num_one_step_observations = 6 + num_dofs * 2 + num_actions + 3
        num_privileged_obs = 6 + num_dofs * 2 + num_actions  + 3

        num_observations = num_actor_history * num_one_step_observations

        env_spacing = 3.  # not used with heightfields/trimeshes 
        send_timeouts = True # send time out information to the algorithm
        episode_length_s = 30 # episode length in seconds

        tarSpeed = 1.0
        tarChangeStepsMin = 3
        tarChangeStepsMax = 5 #每个epoch 保持固定

        tarDistMax = 1.0
        tarHeightMin = 0.3
        tarHeightMax = 0.8


        next_goal_threshold = 0.2
        reach_goal_delay = 0.1
        num_future_goal_obs = 2
        
    class commands(LeggedRobotCfg.commands):
        curriculum = False
        max_curriculum = 2.0
        num_commands = 1 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10. # time before command are changed[s]
        class ranges:
            lin_vel = [-0.5, 0.5] # min max [m/s]


    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.8] # x,y,z [m]
        default_joint_angles = {  # = target angles [rad] when action = 0.0
            
            'left_hip_pitch_joint': -0.1, # in use
            'left_hip_roll_joint': 0.0, # in use
            'left_hip_yaw_joint': 0.0, # in use
            'left_knee_joint': 0.3, # in use
            'left_ankle_pitch_joint': -0.2, # in use
            'left_ankle_roll_joint': 0.0,

            'right_hip_pitch_joint': -0.1, # in use
            'right_hip_roll_joint': 0.0, # in use
            'right_hip_yaw_joint': 0.0, # in use
            'right_knee_joint': 0.3, # in use
            'right_ankle_pitch_joint': -0.2, # in use
            'right_ankle_roll_joint': 0.0,

            'waist_yaw_joint': 0.0, # in use

            'left_shoulder_pitch_joint': 0.0, # in use
            'left_shoulder_roll_joint': 0.1, # in use
            'left_shoulder_yaw_joint': 0.0, # in use
            'left_elbow_joint': 1.2, # in use
            'left_wrist_roll_joint': 0.0, # in use


            'right_shoulder_pitch_joint': 0.0, # in use
            'right_shoulder_roll_joint': -0.1,  # in use
            'right_shoulder_yaw_joint': 0.0, # in use
            'right_elbow_joint': 1.2,  # in use
            'right_wrist_roll_joint': 0.0, # in use
            }



    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        control_type = 'P'
          # PD Drive parameters:
        stiffness = {'hip_yaw': 150,
                     'hip_roll': 150,
                     'hip_pitch': 150,
                     'knee': 300,
                     'ankle': 40,
                     'shoulder': 150,
                     'elbow': 150,
                     'waist_yaw': 150,
                     'wrist': 20,
                     }  # [N*m/rad]
        damping = {  'hip_yaw': 2,
                     'hip_roll': 2,
                     'hip_pitch': 2,
                     'knee': 4,
                     'ankle': 2,
                     'shoulder': 2,
                     'elbow': 2,
                     'waist_yaw': 2,
                     'wrist': 0.5,
                     }  # [N*m/rad]  # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4
        curriculum_joints = ['waist_yaw_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint']
        left_leg_joints = ['left_hip_yaw_joint', 'left_hip_roll_joint', 'left_hip_pitch_joint', 'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint']
        right_leg_joints = ['right_hip_yaw_joint', 'right_hip_roll_joint', 'right_hip_pitch_joint', 'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint']

        left_arm_joints = ['left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint', 'left_elbow_joint', 'left_wrist_roll_joint']
        right_arm_joints = ['right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint', 'right_elbow_joint', 'right_wrist_roll_joint']
        upper_body_link = "pelvis"  # "torso_link"

        left_hip_joints = ['left_hip_yaw_joint', 'left_hip_roll_joint', 'left_hip_pitch_joint']
        right_hip_joints = ['right_hip_yaw_joint', 'right_hip_roll_joint', 'right_hip_pitch_joint']


    class terrain:
        mesh_type = 'plane' # "heightfield" # none, plane, heightfield or trimesh
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
        dynamic = False
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.
        measure_heights = False
        measure_interval = 5
        height_buffer_len = 4
        measure_horizontal_noise = 0.0
        selected = False # select a unique terrain type and pass all arguments
        terrain_kwargs = None # Dict of arguments for selected terrain
        max_init_terrain_level = 5 # starting curriculum state
        terrain_length = 18.
        terrain_width = 4
        num_rows= 10 # number of terrain rows (levels)  # spreaded is benifitiall !
        num_cols =40 # number of terrain cols (types)
        terrain_dict = {"parkour_flat": 1.0,
                        "parkour_hurdle": 0.0,
                        "parkour_wall": 0.0,
                        "parkour_step": 0.0,
                        "parkour_gap": 0.0}

        terrain_proportions = list(terrain_dict.values())
        
        # trimesh only:
        slope_treshold = 1.5# slopes above this threshold will be corrected to vertical surfaces
        origin_zero_z = True

        num_goals = 8


    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/g1/urdf/g1_29dof_rev_1_0_allcollision.urdf'
        marker = '{LEGGED_GYM_ROOT_DIR}/resources/gymassets/mjcf/location_marker.urdf'
        name = "g1"
        foot_name = "ankle_pitch"
        left_foot_name = "left_foot"
        right_foot_name = "right_foot"
        penalize_contacts_on = ["hip", "knee", "torso", "shoulder", "elbow", "pelvis", "hand"]
        terminate_after_contacts_on = []

        waist_joints = ["waist_yaw_joint"]
        knee_joints = ['left_knee_joint', 'right_knee_joint']
        ankle_joints = [ "left_ankle_pitch_joint", "right_ankle_pitch_joint", "left_wrist_roll_joint", "right_wrist_roll_joint"]
        upper_body_link = "torso_link"
        imu_link = "imu_link"
        knee_names = ["left_knee_link", "right_knee_link"]
        
        disable_gravity = False
        collapse_fixed_joints = False # merge bodies connected by fixed joints. Specific fixed joints can be kept by adding " <... dont_collapse="true">
        fix_base_link = False # fixe the base of the robot
        default_dof_drive_mode = 3 # see GymDofDriveModeFlags (0 is none, 1 is pos tgt, 2 is vel tgt, 3 effort)
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        replace_cylinder_with_capsule = True # replace collision cylinders with capsules, leads to faster/more stable simulation
        flip_visual_attachments = False

        density = 0.001
        angular_damping = 0.01
        linear_damping = 0.01
        max_angular_velocity = 1000.
        max_linear_velocity = 1000.
        armature = 0.01
        thickness = 0.01

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
        push_interval_s = 15
        max_push_vel_xy = 1.5

        delay = True

        
    class rewards:
        class scales:
            
            # tracking rewards
            base_height = -10.0
            reach = 5.0 #大一点
            constraint_foot_vel = 0.1 #scale 小一点
            # regularization rewards
            # lin_vel_z = -1.0
            ang_vel_xy = -0.05
            orientation = -2.
            dof_acc = -2.5e-7
            # collision = -15.
            action_rate = -0.3
            delta_torques = -1.0e-7
            torques = -1e-5 # -1e-4
            dof_vel = -5e-4
            feet_air_time = 1.0  #允许脚步轻微移动 # 桌子距离变化丰富一点

            dof_pos_limits = -2.0 #-2.0
            dof_vel_limits = -1.0 #0.0
            torque_limits = -1.0 #0.0

            # deviation_all_joint = -0.0
            # deviation_arm_joint = -0.5
            # deviation_leg_joint = -0.0
            # deviation_hip_joint = -0.5 #base——height 低一点 sacle 增大
            # deviation_waist_joint = -0.0
            # deviation_ankle_joint = -0.0
            
            # feet_distance_lateral = 0.5
            # feet_ground_parallel = -0.01
            # feet_parallel = -1e-2
            
            # feet_clearance = -1.0
            # feet_ground_parallel = -0.015

        only_positive_rewards = False # if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)
        soft_dof_pos_limit = 0.975 # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 0.9
        soft_torque_limit = 0.95
        base_height_target = 0.728
        max_contact_force = 350. # forces above this value are penalized
        clearance_height_target = -0.66
        least_feet_distance = 0.18
        least_feet_distance_lateral = 0.18
        least_knee_distance_lateral = 0.18

class G1CfgPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
    class runner( LeggedRobotCfgPPO.runner ):
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 100 # per iteration
        max_iterations = 200000 # number of policy updates

        # logging
        save_interval = 100 # check for potential saves every this many iterations
        run_name = 'HIM'
        experiment_name = 'g1'
        wandb_project = "pureg1"
        logger = 'wandb'
        
        # load and resume
        resume = False
        load_run = -1 # -1 = last run
        checkpoint = -1 # -1 = last saved model
        resume_path = None # updated from load_run and chkpt