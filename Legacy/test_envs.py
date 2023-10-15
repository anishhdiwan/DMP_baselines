import isaacgym
import isaacgymenvs
import torch


num_envs = 3


envs = isaacgymenvs.make(
	seed=0, 
	task="Cartpole", 
	num_envs=num_envs, 
	sim_device="cuda:0",
	rl_device="cuda:0",
	graphics_device_id=0,
	headless=False,
	multi_gpu=False,
	virtual_screen_capture=False,
	force_render=False,
)

envs.is_vector_env = True

print("Observation space is", envs.observation_space)
print("Action space is", envs.action_space)
obs = envs.reset()
for i in range(10):
	random_actions = 2.0 * torch.rand((num_envs,) + envs.action_space.shape, device = 'cuda:0') - 1.0
	# random_actions = random_actions * 0.0
	envs.step(random_actions)
	print(random_actions)
	print(random_actions.shape)
	# print(envs.obs_dict["obs"])
	print(envs.rew_buf)
	print(envs.rew_buf.shape)
	print("---")
	# if(torch.count_nonzero(envs.reset_buf) > 0):
	# 	print(f"sim step {i} reset {envs.reset_buf}")
	# 	print("---")
	envs.render(mode="rgb_array")
