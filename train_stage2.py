import logging

from vector_quants.trainer import ARTrainer
from vector_quants.utils import get_cfg

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("vq")


def main():
    cfg = get_cfg()

    trainer = ARTrainer(cfg)
    if cfg.train.eval_only:
        trainer.eval()
    else:
        trainer.train()


if __name__ == "__main__":
    main()
