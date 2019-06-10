random:
	python -m run_random --alg=a2c --env=FlowSched-v0 --network=mlp --num_timesteps=2e1 --play

normal:
	python -m run --alg=a2c --env=FlowSched-v0 --network=mlp --num_timesteps=2e5 --play

bandits:
	python -m run_bandits --alg=a2c --env=FlowSched-v0 --network=mlp --num_timesteps=2e1 --play

fix:
	python -m run_fix_epi --alg=a2c --env=FlowSched-v0 --network=mlp --num_timesteps=2e1 --play


randomMP:
	python -m run_random --alg=a2c --env=FlowSchedMP-v0 --network=mlp --num_timesteps=2e1 --play

normalMP:
	python -m run_multi_path --alg=a2c --env=FlowSchedMP-v0 --network=mlp --num_timesteps=2e5 --play

banditsMP:
	python -m run_bandits --alg=a2c --env=FlowSchedMP-v0 --network=mlp --num_timesteps=2e1 --play

fixMP:
	python -m run_fix_epi --alg=a2c --env=FlowSchedMP-v0 --network=mlp --num_timesteps=2e1 --play
