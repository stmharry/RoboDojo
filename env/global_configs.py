import os

from utils.storage import assets_root

current_file = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file)

ROOT_DIR = os.path.join(current_dir, "..")

BENCHMARK = "RoboDojo"
ASSETS_PATH = str(assets_root())
OBJECTS_PATH = os.path.join(ASSETS_PATH, "Object", BENCHMARK)
ROBOTS_PATH = os.path.join(ASSETS_PATH, "Robots")
ENV_CONFIG_PATH = os.path.join(ROOT_DIR, "env_cfg")

ENV_REGEX_NAMESPACE = "{ENV_REGEX_NS}"

BATCH_NUM = 10

USD_PATH = os.environ.get("ROBODOJO_USD_ASSET_PREFIX") or None
