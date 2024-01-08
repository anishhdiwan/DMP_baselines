import isaacgym
import isaacgymenvs
import torch


class isaacgymCEM():
	'''
	An implementation of the cross entropy method for optimization that exploits the multiple env workflow of Isaacgym
	'''

	def __init__(self, num_envs=1000, device="cuda:0", cem_iterations=5):
		self.cem_iterations = cem_iterations
		self.envs = self.init_envs(num_envs=num_envs, device=device)
		self.init_mu = 0.0
		self.init_sigma = 1.0
		self.mu = None
		self.sigma = None
		self.elite_fraction = 0.4
		# obs = envs.reset()



	def init_envs(self, num_envs=100, device="cuda:0"):
		'''
		Initialise num_envs number of cartpole environments with the given device. 
		'''

		envs = isaacgymenvs.make(
			seed=0, 
			task="Cartpole", 
			num_envs=num_envs, 
			sim_device=device,
			rl_device=device,
			graphics_device_id=0,
			headless=False,
			multi_gpu=False,
			virtual_screen_capture=False,
			force_render=False,
		)

		envs.is_vector_env = True
		print("Observation space is", envs.observation_space)
		print("Action space is", envs.action_space)

		return envs

	

	def sample_and_evaluate_actions(self):
		'''
		Evaluate the sampled actions under the dynamics and reward model via simulation. i.e, Apply a sampled action to each env to get a reward signal
		'''
		actions = torch.normal(self.init_mu, self.init_sigma, size=(self.num_envs, 1))
		self.envs.step(actions)
		return actions, self.envs.rew_buf

	def update_sampling_dist(self, actions, evals):
		'''
		Use the elite solutions in the population (of actions) to update the sampling normal distribution
		'''
		sorted_evals, indices = torch.sort(evals)
		sorted_actions = actions[indices]
		self.mu = torch.mean(sorted_actions[:(int(self.elite_fraction*self.num_envs))], dim=0)
		self.sigma = torch.std(sorted_actions[:(int(self.elite_fraction*self.num_envs))], dim=0)

	def cem(self):
		'''
		Combine sampling, evaluation, and distribution updates
		'''
		for _ in range(self.cem_iterations):
			actions, rewards = self.sample_and_evaluate_actions()
			self.update_sampling_dist(actions, rewards)


		# TO DO: is the updated sampling distribution conditioned on the current state? If so, how to apply the sample from the updated
		#        distribution to the real env? Won't this require resetting the envs to the next state of the real env?








obs = envs.reset()
for i in range(2000):
	random_actions = 2.0 * torch.rand((num_envs,) + envs.action_space.shape, device = 'cuda:0') - 1.0
	# random_actions = random_actions * 0.0
	envs.step(random_actions)
	print(envs.obs_dict["obs"])
	print(envs.rew_buf)
	print("----")
	# if(torch.count_nonzero(envs.reset_buf) > 0):
	# 	print(f"sim step {i} reset {envs.reset_buf}")
	# 	print("---")
	envs.render(mode="rgb_array")
