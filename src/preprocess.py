import argparse
import joblib
import logging
import numpy as np
from pathlib import Path
from sgfmill import sgf
from sgfmill import sgf_moves
from sgfmill import ascii_boards
import time
from tqdm import tqdm
import yaml

from game_record import GoGameRecord
from model_config import ModelConfig

logging_format = '%(asctime)s %(module)s:%(lineno)d %(funcName)s %(levelname)s %(message)s'


def sgf2record():
    model_config = ModelConfig()
    sgf_path = Path(args.in_root)
    out_path = Path(args.out_root)
    out_path.mkdir(exist_ok=True, parents=True)
    assert sgf_path.exists()
    sgf_files = [sgf_file for sgf_file in sgf_path.glob('./**/*.sgf')]
    print(f'processing {len(sgf_files)} SGF files in {sgf_path}')
    board_size = model_config.board_size
    num_saved = 0
    indices = []
    record_size = GoGameRecord.record_size(model_config)
    for sgf_file in tqdm(sgf_files):
        with open(sgf_file, 'rb') as in_fd:
            sgf_src = in_fd.read()
            try:
                sgf_game = sgf.Sgf_game.from_bytes(sgf_src)
                assert sgf_game.get_size() == board_size
            except ValueError:
                logging.debug(f'bad sgf file {sgf_file}')
                continue
            try:
                board, plays = sgf_moves.get_setup_and_moves(sgf_game)
            except ValueError as e:
                logging.debug(f'fail to parse {sgf_file}: {e}')
                continue
            out_data = np.zeros((len(plays), record_size), dtype=np.int8)
            valid_play = True
            for i, (colour, move) in enumerate(plays):
                if move is None:
                    continue
                # N * N are row x column  +
                out_data[i, :board_size**2] = [GoGameRecord.color_to_player(c)
                                               for r in board.board for c in r]
                row, col = move
                out_data[i, model_config.game_record_row_index] = row
                out_data[i, model_config.game_record_col_index] = col
                out_data[i, model_config.game_record_player_index] = GoGameRecord.color_to_player(colour)
                try:
                    board.play(row, col, colour)
                except ValueError:
                    logging.debug(f"illegal move in sgf file {sgf_file}")
                    valid_play = False
                    break
            if not valid_play:
                continue
            out_file = out_path/f'{sgf_file.stem}.joblib'
            record = GoGameRecord(model_config, out_data)
            record.write_to_file(out_file)
            num_saved += 1
            indices.append((str(out_file), len(plays)))
    index_file = out_path/model_config.game_index_file
    with open(index_file, 'w') as out_fd:
        yaml.dump(indices, out_fd)
    average_plays = sum(l for _, l in indices)/num_saved
    print(f'successfully saved {num_saved} out of {len(sgf_files)} SGF files, {average_plays}')


def show_record():
    game_file = Path(args.in_file)
    # show last step only
    game_record = GoGameRecord.from_file(config=ModelConfig(), in_file=game_file)
    game_record.render_result()


def show_sgf():
    sgf_file = Path(args.in_file)
    assert sgf_file.exists(), sgf_file
    with open(sgf_file, 'rb') as in_fd:
        sgf_src = in_fd.read()
        row, col, color = -1, -1, None
        try:
            sgf_game = sgf.Sgf_game.from_bytes(sgf_src)
        except ValueError:
            raise ValueError(f'bad sgf file {sgf_file}')
        try:
            board, plays = sgf_moves.get_setup_and_moves(sgf_game)
        except ValueError as e:
            raise ValueError(f'fail to parse {sgf_file}: {e}')
        for i, (colour, move) in enumerate(plays):
            if move is None:
                continue
            row, col = move
            try:
                board.play(row, col, colour)
            except ValueError:
                raise ValueError(f"illegal move in sgf file {sgf_file}")
        print(ascii_boards.render_board(board))
        print(f'action=({row}, {col}), player={colour}, score={board.area_score()}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-l', dest='log_level', default='info', help="logging level: {debug, info, error}")
    subparser = parser.add_subparsers(dest='command', title='sub_commands', description='valid commands')
    subparser.required = True

    sub_parser = subparser.add_parser('sgf2record')
    sub_parser.add_argument('in_root', help='root path for input sgf')
    sub_parser.add_argument('out_root', help='root path for out joblib')
    sub_parser.set_defaults(func=sgf2record)

    sub_parser = subparser.add_parser('show_record')
    sub_parser.add_argument('in_file', help='path for game record')
    sub_parser.set_defaults(func=show_record)

    sub_parser = subparser.add_parser('show_sgf')
    sub_parser.add_argument('in_file', help='path for sgf')
    sub_parser.set_defaults(func=show_sgf)

    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format=logging_format)
    t0 = time.time()
    if args.command:
        args.func()
        logging.info("%s takes %f s" % (args.command, time.time()-t0))
    else:
        parser.print_help()
