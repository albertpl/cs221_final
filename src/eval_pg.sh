#!/usr/bin/env bash

work_root=/tmp/test_pg
weight_file=checkpoint.hdf5
simulate_program=${HOME}/src/cs221_final/src/evaluate.py
game_result_root=${HOME}/src/tmp/game_result/
num_games=1000
start_iter=1
end_iter=6

for i in $(seq ${start_iter} ${end_iter})
do
    current_path=${work_root}/${i}/
    echo ${current_path}
    if [[ ! -d ${current_path} ]]; then
        continue
    fi
    python ${simulate_program} \
        --black_player_record_path ${current_path}/train/worker_1 \
        --weight_root ${current_path}  \
        eval  --result_root ${game_result_root} \
        nn random ${num_games}
done