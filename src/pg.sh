#!/usr/bin/env bash

work_root=/tmp/test_pg
num_iter=3
prev_path=/tmp/test_pg/iter_1/
tensor_log_dir=/tmp/log_dir
weight_file=checkpoint.hdf5
simulate_program=${HOME}/src/cs221_final/src/evaluate.py
train_program=${HOME}/src/cs221_final/src/pipeline.py
num_games=5000
train_epochs=20

for i in $(seq ${num_iter})
do
    current_path=${work_root}/${i}/
    mkdir -p ${current_path}
    echo ${current_path}
    # copy weight from last iteration
    cp ${prev_path}/${weight_file} ${current_path}/
    # start gather experience via simulations
    python ${simulate_program} \
        --black_player_record_path ${current_path}/train/worker_1 \
        --weight_root ${current_path}  \
        eval  nn random ${num_games} &
    wait
    #
    python ${train_program} \
        --model_name policy_gradient\
        --training_epochs ${train_epochs} \
        train  \
        --log_dir ${tensor_log_dir} \
        ${current_path} ${current_path}
    prev_path=${current_path}
done