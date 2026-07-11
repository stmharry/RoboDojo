import os

from robodojo.core.paths import discover_repository_root
from robodojo.core.storage import assets_root

ROOT_DIR = str(discover_repository_root())

BENCHMARK = "RoboDojo"
ASSETS_PATH = str(assets_root())
OBJECTS_PATH = os.path.join(ASSETS_PATH, "Object", BENCHMARK)
ROBOTS_PATH = os.path.join(ASSETS_PATH, "Robots")
ENV_CONFIG_PATH = os.path.join(ROOT_DIR, "configs")

ENV_REGEX_NAMESPACE = "{ENV_REGEX_NS}"

BATCH_NUM = 10

USD_PATH = os.environ.get("ROBODOJO_USD_ASSET_PREFIX") or None
