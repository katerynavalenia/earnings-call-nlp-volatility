"""
Master pipeline orchestrator.

Runs all 12 steps in sequence: parsing, labeling, training,
prediction, volatility computation, regression, and visualization.
"""

import argparse
import importlib
import logging
import os
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

# Pipeline steps: (step_number, label, module_path)
STEPS = [
    (1,  "Parse Transcripts",           "src.preprocessing.parse_transcripts"),
    (2,  "Map Tickers",                 "src.preprocessing.ticker_mapping"),
    (3,  "Lexicon Scoring",             "src.labeling.lexicon_labeler"),
    (4,  "Build Training Set",          "src.labeling.build_training_set"),
    (5,  "Train FinBERT Model",         "src.modeling.train_uncertainty_model"),
    (6,  "Evaluate Model",              "src.modeling.evaluate_model"),
    (7,  "Predict Uncertainty",         "src.modeling.predict_uncertainty"),
    (8,  "Compute Volatility",          "src.features.compute_volatility"),
    (9,  "Build Panel Dataset",         "src.features.build_panel"),
    (10, "Main Regression",             "src.analysis.regression"),
    (11, "Robustness Checks",           "src.analysis.robustness"),
    (12, "Visualizations",              "src.analysis.visualizations"),
]


def run_step(num, label, module_path):
    """Import and run one pipeline step."""
    log.info("=" * 60)
    log.info("STEP %d/%d: %s", num, len(STEPS), label)
    log.info("=" * 60)

    t0 = time.time()
    # Clear sys.argv so sub-module argparse doesn't see run_pipeline's args
    saved_argv = sys.argv
    sys.argv = [module_path]
    try:
        mod = importlib.import_module(module_path)
        mod.main()
    except SystemExit as e:
        if e.code and e.code != 0:
            log.error("Step %d exited with code %s", num, e.code)
            return False
    except Exception:
        log.exception("Step %d failed", num)
        return False
    finally:
        sys.argv = saved_argv

    elapsed = time.time() - t0
    log.info("Step %d completed in %.1f s", num, elapsed)
    return True


def main():
    parser = argparse.ArgumentParser(description="Run the full analysis pipeline.")
    parser.add_argument(
        "--from", dest="from_step", type=int, default=1,
        help="Resume from this step number (inclusive)."
    )
    parser.add_argument(
        "--only", nargs="+", type=int, default=None,
        help="Run only these step numbers."
    )
    parser.add_argument(
        "--skip-training", action="store_true",
        help="Skip steps 5-6 (model training/evaluation). "
             "Assumes a trained model exists in models/uncertainty_finbert/."
    )
    args = parser.parse_args()

    if args.only:
        step_nums = set(args.only)
    else:
        step_nums = set(range(args.from_step, len(STEPS) + 1))

    if args.skip_training:
        step_nums -= {5, 6}

    log.info("Pipeline starting. Steps to run: %s", sorted(step_nums))
    total_t0 = time.time()
    failed = []

    for num, label, module_path in STEPS:
        if num not in step_nums:
            log.info("Skipping step %d: %s", num, label)
            continue
        ok = run_step(num, label, module_path)
        if not ok:
            failed.append(num)
            log.error("Stopping pipeline due to failure at step %d.", num)
            break

    elapsed = time.time() - total_t0
    log.info("=" * 60)
    if failed:
        log.error(
            "Pipeline FAILED at step(s) %s (total time: %.0f s).", failed, elapsed
        )
        sys.exit(1)
    else:
        log.info("Pipeline COMPLETE (total time: %.0f s).", elapsed)


if __name__ == "__main__":
    main()
