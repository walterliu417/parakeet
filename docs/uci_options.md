**List of UCI options**\
\
explore_factor - How much the engine explores moves during the PUCT search. Setting too high could cause the engine to waste time searching useless moves, while setting too low might make the engine miss moves.\
capture_bonus - How much the engine explores captures during PUCT.\
check_bonus - How much the engine explores checking moves during PUCT.\
explore_decay - A value above 100 forces the engine to conclude exploration before the search time limit and only focus on the best moves during each rollout.\
tablebase_dir - Path to 5-piece Syzygy tablebases.\
net_path - Path to the neural network, stored as an Open Neural Network Exchange (ONNX) file.\
\
See technical.md for explanations on the search algorithm.\