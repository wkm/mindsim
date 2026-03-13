import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from botcad.packing import solve_packing
from bots.wheeler_base.design import build

bot = build()
solve_packing(bot)

base = bot.root
print(f"Base solved dimensions: {base.solved_dimensions}")
print(f"Base height explicit? {base.explicit_dimensions}, height prop: {base.height}")


def _check():
    import warnings

    with warnings.catch_warnings(record=True) as w:
        bot.solve()
        for warn in w:
            print(f"WARNING: {warn.message}")


_check()
