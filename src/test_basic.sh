#!/usr/bin/env bash

TMP_PATH=/tmp/test_save_record/
WEIGHT_PATH=/tmp/test_save_record/

function show_records()
{
    for i in ${TMP_PATH}/*.joblib
    do
        echo ${i}
        python ~/src/cs221_final/src/preprocess.py show_record ${i}
    done
}

for player1 in random mcts nn
do
    for player2 in random nn pachi
    do
        echo ${player1} ${player2}
        rm -fr ${TMP_PATH}
        python ${HOME}/src/cs221_final/src/evaluate.py  \
            --mcts_num_rollout 10 \
            --weight_root ${WEIGHT_PATH} \
            --black_player_record_path ${TMP_PATH} \
            --white_player_record_path ${TMP_PATH} \
            --print_board=1 \
            eval \
            ${player1} ${player2} 1
        show_records
    done
done


