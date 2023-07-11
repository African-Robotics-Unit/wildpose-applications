import hjson


def load_config_file(fpath: str):
    config = None
    with open(fpath, mode='r') as f:
        config = hjson.loads(f.read())
    return config
