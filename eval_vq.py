import logging

from vector_quants.evaluator import VQEvaluator
from vector_quants.utils import get_cfg

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("vq")


def main():
    cfg = get_cfg()

    evaluator = VQEvaluator(cfg)
    evaluator.eval()


if __name__ == "__main__":
    main()
