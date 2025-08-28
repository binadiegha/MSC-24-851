import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).parent))

from .agent import Agent

from . import parallel_parking

TASKS = globals().get("TASKS", {})
TASKS.update({
"parallel_parking": parallel_parking.make,
})