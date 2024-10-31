import logging

from vector_quants.evaluator import AREvaluator
from vector_quants.utils import get_cfg

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("vq")


def main():
    cfg = get_cfg()

    evaluator = AREvaluator(cfg)

    if cfg.sample.sample_file:
        evaluator.eval_openai()
    else:
        evaluator.eval()


if __name__ == "__main__":
    main()
