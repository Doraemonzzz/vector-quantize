import logging

from vector_quants.trainer import VQTrainer
from vector_quants.utils import get_cfg

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("vq")
import os
import traceback


# see https://github.com/wandb/wandb/issues/4929
# need add this, otherwise wandb may get stuck
def exit():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts/kill.sh")
    os.system(f"bash {path}")


def main():
    cfg = get_cfg()

    trainer = VQTrainer(cfg)
    if cfg.train.eval_only:
        trainer.eval()
    else:
        trainer.train()


if __name__ == "__main__":
    try:
        main()
        exit()
    except Exception:
        traceback.print_exc()
        exit()
    finally:
        exit()
