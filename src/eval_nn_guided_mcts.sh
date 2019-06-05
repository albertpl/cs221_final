#!/usr/bin/env bash

work_root=/tmp/test_nn_guided_mcts
weight_file=checkpoint.hdf5
simulate_program=${HOME}/src/cs221_final/src/evaluate.py
game_result_root=${HOME}/src/tmp/game_result/
num_games=1000
start_iter=1
end_iter=5
mcts_num_rollout=100

for i in $(seq ${start_iter} ${end_iter})
do
    current_path=${work_root}/${i}/
    echo ${current_path}
    if [[ ! -d ${current_path} ]]; then
        continue
    fi
    python ${simulate_program} \
        --mcts_num_rollout ${mcts_num_rollout} \
        --weight_root ${current_path}  \
        eval  --result_root ${game_result_root} \
        nn_guided_mcts random ${num_games}
done