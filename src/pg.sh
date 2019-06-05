#!/usr/bin/env bash

work_root=/tmp/test_pg
start_iter=1
end_iter=20
prev_path=/tmp/test_pg/start/
tensor_log_dir=/tmp/log_dir
weight_file=checkpoint.hdf5
simulate_program=${HOME}/src/cs221_final/src/evaluate.py
train_program=${HOME}/src/cs221_final/src/pipeline.py
num_games=1000
train_epochs=2

for i in $(seq ${start_iter} ${end_iter})
do
    current_path=${work_root}/${i}/
    mkdir -p ${current_path}
    echo ${current_path}
    # copy weight from last iteration
    cp ${prev_path}/${weight_file} ${current_path}/
    # gathering experience via simulations
    python ${simulate_program} \
        --black_player_record_path ${current_path}/train/worker_1 \
        --weight_root ${current_path}  \
        eval  nn pachi ${num_games}
    #python ${simulate_program} \
    #    --black_player_record_path ${current_path}/train/worker_2 \
    #    --weight_root ${current_path}  \
    #    eval  nn pachi ${num_games} &
    #wait
    # training with experiences
    python ${train_program} \
        --model_name policy_gradient\
        --training_epochs ${train_epochs} \
        train  \
        --log_dir ${tensor_log_dir} \
        ${current_path} ${current_path}
    prev_path=${current_path}
done