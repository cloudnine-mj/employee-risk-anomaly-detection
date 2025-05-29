import yaml
from typing import Any, Dict


def load_config(path: str) -> Dict[str, Any]:
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Configuration loaded from {path} is not a dict")
    return cfg

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Load and print config')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    args = parser.parse_args()
    config = load_config(args.config)
    print(yaml.dump(config, sort_keys=False))
