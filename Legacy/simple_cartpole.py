import os
from isaacgym import gymapi
from isaacgym import gymutil

# initialize gym
gym = gymapi.acquire_gym()

# parse arguments
args = gymutil.parse_arguments(description="Asset and Environment Information")

# create simulation context
sim_params = gymapi.SimParams()

sim_params.substeps = 1
sim_params.physx.solver_type = 1
sim_params.physx.num_position_iterations = 4
sim_params.physx.num_velocity_iterations = 1
sim_params.physx.num_threads = args.num_threads
sim_params.physx.use_gpu = args.use_gpu

sim_params.use_gpu_pipeline = False
if args.use_gpu_pipeline:
    print("Using CPU pipeline. Change sim_params.use_gpu_pipeline to use something else")


sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)

cam_props = gymapi.CameraProperties()
viewer = gym.create_viewer(sim, cam_props)

plane_params = gymapi.PlaneParams()
gym.add_ground(sim, plane_params)

asset_root = "./assets"
cartpole_asset = "cartpole.urdf"
asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = True

# num_envs = 12
# assets = []
# for i in range(num_envs):
# 	current_asset = gym.load_asset(sim, asset_root, cartpole_asset, asset_options)
# 	assets.append(current_asset)

asset = gym.load_asset(sim, asset_root, cartpole_asset, asset_options)

spacing = 5.0
lower = gymapi.Vec3(-spacing, 0.0, -spacing)
upper = gymapi.Vec3(spacing, spacing, spacing)
env = gym.create_env(sim, lower, upper, 5)

pose = gymapi.Transform()
pose.p = gymapi.Vec3(0.0, 1.0, 0.0)
pose.r = gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107)

actor_handle = gym.create_actor(env, asset, pose, "MyActor", 0, 1)


props = gym.get_actor_dof_properties(env, actor_handle)
props["driveMode"].fill(gymapi.DOF_MODE_NONE)
props["stiffness"].fill(0.0)
props["damping"].fill(0.0)
gym.set_actor_dof_properties(env, actor_handle, props)


gym.viewer_camera_look_at(viewer, None, gymapi.Vec3(20, 5, 20), gymapi.Vec3(0, 1, 0))


while not gym.query_viewer_has_closed(viewer):

    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # update the viewer
    gym.step_graphics(sim);
    gym.draw_viewer(viewer, sim, True)

    # Wait for dt to elapse in real time.
    # This synchronizes the physics simulation with the rendering rate.
    gym.sync_frame_time(sim)


gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
