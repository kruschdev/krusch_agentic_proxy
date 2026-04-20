import platform
from pathlib import Path
import sys
from importlib.machinery import ModuleSpec

if 'mlx' not in sys.modules:
    sys.modules['mlx'] = type('MockMlx', (), {'__spec__': ModuleSpec('mlx', None)})()
if 'mlx.core' not in sys.modules:
    sys.modules['mlx.core'] = type('MockMlxCore', (), {'__spec__': ModuleSpec('mlx.core', None)})()
if 'mlx_lm' not in sys.modules:
    sys.modules['mlx_lm'] = type('MockMlxLm', (), {'__spec__': ModuleSpec('mlx_lm', None)})()

HOME_DIR = Path.home()
REPO_ROOT = Path(__file__).parent.parent.resolve()
DEFAULT_HF_HOME = f"{HOME_DIR}/.cache/huggingface/hub"

FLAG_ON_MAC = platform.system() == "Darwin"
FLAG_ON_LINUX = platform.system() == "Linux"
