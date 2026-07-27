"""
QuantaAlpha CLI entry.

Commands:
  quantaalpha mine       - run factor mining
  quantaalpha backtest   - run backtest
  quantaalpha ui         - start log Web UI
  quantaalpha health_check - environment health check
"""

from pathlib import Path
from dotenv import load_dotenv

# Load .env (prefer project root, fallback to cwd)
_project_root = Path(__file__).resolve().parents[1]
_env_path = _project_root / ".env"
if _env_path.exists():
    load_dotenv(_env_path)
else:
    load_dotenv(".env")

import fire
from quantaalpha.pipeline.factor_mining import main as mine
from quantaalpha.pipeline.factor_backtest import main as backtest
from quantaalpha.app.utils.health_check import health_check
from quantaalpha.app.utils.info import collect_info

# The one place global RNG state is seeded. `configs/backtest.yaml:11
# random_seed` is dead -- nothing under quantaalpha/backtest/ consumes it -- so
# this is the real entry point. The evolution operators use unseeded
# `random.shuffle`/`random.sample`/`random.choices`
# (crossover.py:370/412/432/502, mutation.py:217); seeding here makes those
# reproducible. Property 2 binds E_theta, not the generator, but a clean A/B
# needs both arms to explore identically.
# (Note: TrajectoryPool.select_parents_for_crossover at trajectory.py:251 also
# shuffles, but is dead code -- the controller uses
# crossover_op.select_crossover_pairs.)
_DEFAULT_SEED = 42


def _seed_everything() -> int:
    import os
    import random

    import numpy as np

    seed = os.environ.get("QA_SEED")
    if seed is None:
        try:
            from quantaalpha.pipeline.planning import load_run_config

            config_path = os.environ.get("QA_CONFIG") or (
                _project_root / "configs" / "experiment.yaml"
            )
            seed = (load_run_config(Path(config_path)) or {}).get("seed")
        except Exception:
            seed = None
    try:
        seed = int(seed) if seed is not None else _DEFAULT_SEED
    except (TypeError, ValueError):
        seed = _DEFAULT_SEED

    random.seed(seed)
    np.random.seed(seed)
    return seed


def app():
    seed = _seed_everything()
    print(f"[quantaalpha] RNG seed = {seed}")
    fire.Fire(
        {
            "mine": mine,
            "backtest": backtest,
            "health_check": health_check,
            "collect_info": collect_info,
        }
    )


if __name__ == "__main__":
    app()
