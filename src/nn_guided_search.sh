#!/usr/bin/env bash

tensor_log_dir=/tmp/log_dir
weight_file=checkpoint.hdf5
simulate_program=${HOME}/src/cs221_final/src/evaluate.py
train_program=${HOME}/src/cs221_final/src/pipeline.py
work_root=/tmp/test_nn_guided_mcts
prev_path=${work_root}/32
start_iter=33
end_iter=100
num_games=10
train_epochs=1
mcts_num_rollout=1000
num_workers=1
c_puct=1.0
mcts_dirichlet_alpha=0.13

#python ${train_program} \
#    --model_name supervised \
#    --training_epochs 1 \
#    train \
#    --log_dir /tmp/log_dir/ \
#    /tmp/test_nn_guided_mcts/start/ /tmp/test_nn_guided_mcts/start/ \

for i in $(seq ${start_iter} ${end_iter})
do
    current_path=${work_root}/${i}/
    mkdir -p ${current_path}
    echo ${current_path}
    # copy weight from last iteration
    cp ${prev_path}/${weight_file} ${current_path}/
    # start gather experience via simulations
    for w in $(seq ${num_workers})
    do
        python ${simulate_program} \
            --mcts_dirichlet_alpha ${mcts_dirichlet_alpha} \
            --mcts_c_puct ${c_puct} \
            --mcts_num_rollout ${mcts_num_rollout} \
            --black_player_record_path ${current_path}/train/worker_${w} \
            --white_player_record_path ${current_path}/train/worker_${w} \
            --weight_root ${current_path}  \
            eval  nn_guided_mcts nn_guided_mcts ${num_games} &
    done
    wait
    python ${train_program} \
        --model_name supervised\
        --training_epochs ${train_epochs} \
        train  \
        --log_dir ${tensor_log_dir} \
        ${current_path} ${current_path}
    prev_path=${current_path}
done
