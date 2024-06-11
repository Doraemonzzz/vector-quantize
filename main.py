import logging

from vector_quants.trainer import VQTrainer
from vector_quants.utils import get_args

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("vq")


def main():
    args = get_args()
    trainer = VQTrainer(args)
    if args.eval_only:
        trainer.eval()
    else:
        trainer.train()


if __name__ == "__main__":
    main()
