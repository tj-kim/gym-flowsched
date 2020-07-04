# For single link network with synthetic data
random:
	python -m run_random --alg=a2c --env=FlowSched-v0 --network=mlp --num_timesteps=2e1 --play

normal:
	python -m run --alg=a2c --env=FlowSched-v0 --network=mlp --num_timesteps=2e5 --play

normalNoTrain:
	python -m run --alg=a2c --env=FlowSched-v0 --network=mlp --num_timesteps=2e0 --play

bandits:
	python -m run_bandits --alg=a2c --env=FlowSched-v0 --network=mlp --num_timesteps=2e1 --play

fix:
	python -m run_fix_epi --alg=a2c --env=FlowSched-v0 --network=mlp --num_timesteps=2e1 --play

# For multi-path network with synthetic data
randomMP:
	python -m run_mp_random --alg=a2c --env=FlowSchedMP-v0 --network=mlp --num_timesteps=2e1 --play

normalMP:
	python -m run_multi_path --alg=a2c --env=FlowSchedMP-v0 --network=mlp --num_timesteps=2e5 --play

normalMP_no_train:
	python -m run_multi_path --alg=a2c --env=FlowSchedMP-v0 --network=mlp --num_timesteps=2e0 --play

banditsMP:
	python -m run_mp_bandits --alg=a2c --env=FlowSchedMP-v0 --network=mlp --num_timesteps=2e1 --play

fixMP:
	python -m run_mp_fix_epi --alg=a2c --env=FlowSchedMP-v0 --network=mlp --num_timesteps=2e1 --play


# For multi-path network with data read from files
randomData:
	python -m run_mp_random --alg=a2c --env=FlowSchedData-v0 --network=mlp --num_timesteps=2e1 --play

normalData:
	python -m run_multi_path --alg=a2c --env=FlowSchedData-v0 --network=mlp --num_timesteps=2e6 --play

banditsData:
	python -m run_mp_bandits --alg=a2c --env=FlowSchedData-v0 --network=mlp --num_timesteps=2e1 --play

fixData:
	python -m run_mp_fix_epi --alg=a2c --env=FlowSchedData-v0 --network=mlp --num_timesteps=2e2 --play

# For continous actions: sending rate control

normalSR:
	python -m run_multi_path --alg=a2c --env=FlowSchedSdRate-v0 --network=mlp --num_timesteps=2e6 --play

randomSR:
	python -m run_mp_sr_random --alg=a2c --env=FlowSchedSdRate-v0 --network=mlp --num_timesteps=2e0 --play