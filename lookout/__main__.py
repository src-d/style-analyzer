import sys

from lookout.core.cmdline import create_parser


def main():
    parser = create_parser()
    args = parser.parse_args()
    try:
        handler = args.handler
    except AttributeError:
        def print_usage(_):
            parser.print_usage()

        handler = print_usage
    return handler(args)


if __name__ == "__main__":
    sys.exit(main())
