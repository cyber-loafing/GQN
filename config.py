import argparse


def arg_parse():
    parser = argparse.ArgumentParser(description='IM Agent')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    return parser.parse_args()



config = arg_parse()
