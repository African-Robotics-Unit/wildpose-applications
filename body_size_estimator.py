from pprint import pprint

from utils.config import load_config_file


def main():
    config_fpath = 'config.hjson'
    config = load_config_file(config_fpath)
    pprint(config)


if __name__ == '__main__':
    main()
