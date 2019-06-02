#!/usr/bin/env bash

tensor_log_dir=/tmp/log_dir
weight_file=checkpoint.hdf5
simulate_program=${HOME}/src/cs221_final/src/evaluate.py
train_program=${HOME}/src/cs221_final/src/pipeline.py
work_root=/tmp/test_nn_guided_mcts
prev_path=${work_root}/iter_2
num_iter=10
num_games=100
train_epochs=10
mcts_num_rollout=50

for i in $(seq ${num_iter})
do
    current_path=${work_root}/${i}/
    mkdir -p ${current_path}
    echo ${current_path}
    # copy weight from last iteration
    cp ${prev_path}/${weight_file} ${current_path}/
    # start gather experience via simulations
    python ${simulate_program} \
        --mcts_c_puct 1.0 \
        --mcts_num_rollout ${mcts_num_rollout} \
        --black_player_record_path ${current_path}/train/worker_1 \
        --weight_root ${current_path}  \
        eval  nn_guided_mcts random ${num_games}
    # wait
    python ${train_program} \
        --model_name supervised\
        --training_epochs ${train_epochs} \
        train  \
        --log_dir ${tensor_log_dir} \
        ${current_path} ${current_path}
    prev_path=${current_path}
done
