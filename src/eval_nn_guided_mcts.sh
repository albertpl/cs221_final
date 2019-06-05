#!/usr/bin/env bash

work_root=/tmp/test_nn_guided_mcts
weight_file=checkpoint.hdf5
simulate_program=${HOME}/src/cs221_final/src/evaluate.py
game_result_root=${HOME}/src/tmp/game_result/
num_games=10
start_iter=7
end_iter=10
mcts_num_rollout=100
c_puct=0.2

for i in $(seq ${start_iter} ${end_iter})
do
    current_path=${work_root}/${i}/
    echo ${current_path}
    if [[ ! -d ${current_path} ]]; then
        continue
    fi
    python ${simulate_program} \
        --mcts_c_puct ${c_puct} \
        --mcts_num_rollout ${mcts_num_rollout} \
        --weight_root ${current_path}  \
        eval  --result_root ${game_result_root} \
        nn_guided_mcts pachi ${num_games}
done