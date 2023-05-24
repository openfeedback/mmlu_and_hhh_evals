import argparse
from evaluate_decoder_only_model import main
from copy import deepcopy


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrain", "-k", type=int, default=5)
    parser.add_argument("--ngpu", "-g", type=int, default=1)
    parser.add_argument("--data_dir", "-d", type=str, default="data")
    parser.add_argument("--save_dir", "-s", type=str, default="results")
    parser.add_argument("--models", "-m", nargs="+", required=True)

    args = parser.parse_args()
    for model in args.models:
        args_with_one_model = deepcopy(args)
        args_with_one_model.model = model
        main(args_with_one_model)
