
import argparse
from test import test


def parse_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(title="subcommands")
    # 测试
    test_parser = subparsers.add_parser("test")
    test_parser.add_argument("--annotation_file", help="dataset annotation file path", type=str, required=True)
    test_parser.add_argument("--image_root", help="dataset image root path", type=str, required=True)
    test_parser.add_argument('-a', '--arch', type=str, default='resnet50')
    test_parser.add_argument("--checkpoint", help="checkpoint path", type=str, required=True)
    test_parser.add_argument("--batch_size", help="batch size", type=int, default=32)
    test_parser.add_argument("--gpu", help="gpu index, 0, 1, 2, ...", type=int, default=0)
    test_parser.set_defaults(func=test)

    args = parser.parse_args()
    args.func(args)


def main():
    parse_args()


if __name__ == '__main__':
    # test --annotation_file  G:\data_format_transform\new\test.pkl  --image_root  F:\Database\od_dataset\test_format\test_1  --checkpoint ./log/resnet50-bus_id-xent_htri/checkpoint_ep30.pth.tar
    main()
