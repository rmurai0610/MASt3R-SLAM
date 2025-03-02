import re
import yaml

config = {}


def load_config(path="config/base.yaml", is_parent=False):
    # from https://stackoverflow.com/questions/30458977/yaml-loads-5e-6-as-string-and-not-a-number
    loader = yaml.SafeLoader
    loader.add_implicit_resolver(
        "tag:yaml.org,2002:float",
        re.compile(
            """^(?:
                [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
                |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
                |\\.[0-9_]+(?:[eE][-+][0-9]+)?
                |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
                |[-+]?\\.(?:inf|Inf|INF)
                |\\.(?:nan|NaN|NAN))$""",
            re.X,
        ),
        list("-+0123456789."),
    )

    with open(path, "r") as f:
        cfg = yaml.load(f, Loader=loader)
    inherit = cfg.get("inherit")
    if inherit is not None:
        cfg_parent = load_config(inherit, is_parent=True)
    else:
        cfg_parent = dict()
    cfg = merge_config(cfg_parent, cfg)
    if is_parent:
        return cfg

    # update the global config only once at the child node
    set_global_config(cfg)


def merge_config(dict1, dict2):
    for k, v in dict2.items():
        if k not in dict1:
            dict1[k] = dict()
        if isinstance(v, dict):
            merge_config(dict1[k], v)
        else:
            dict1[k] = v
    return dict1


def set_global_config(cfg):
    global config
    config.update(cfg)
    return config
