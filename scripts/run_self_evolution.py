"""
CLI Runner for WebRL-Style Online Curriculum Self-Evolution Loop.

Executes autonomous generations:
1. Curriculum Selection: Chooses level tasks & generates task variants.
2. Online Rollout Collection: Gathers interaction trajectories with past experience injection.
3. First-Class Experience Indexing: Stores compiler/test failure-to-repair pairs into ChromaDB.
4. Held-Out Evaluation & Mastery Check: Evaluates pass rate on held-out tasks.
5. Curriculum Level Promotion: Promotes the model when mastery threshold is achieved (L1 -> L7).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys

from learning.self_evolution import SelfEvolutionController

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("SelfEvolutionRunner")


async def main_async(args: argparse.Namespace) -> None:
    logger.info("=" * 70)
    logger.info("   WebRL-Style Hardware Online Curriculum Self-Evolution Loop")
    logger.info("=" * 70)
    logger.info(f"Start Level:          Level {args.start_level}")
    logger.info(f"Max Level:            Level {args.max_level}")
    logger.info(f"Generations:          {args.generations}")
    logger.info(f"Promotion Threshold:  {args.promotion_threshold * 100:.0f}%")
    logger.info(f"Task Generator:       {'Enabled' if args.enable_task_generator else 'Disabled'}")
    logger.info(f"Working Directory:    {args.work_dir}")
    logger.info("=" * 70)

    controller = SelfEvolutionController(
        start_level=args.start_level,
        max_level=args.max_level,
        promotion_threshold=args.promotion_threshold,
        work_dir=args.work_dir,
    )

    for gen in range(1, args.generations + 1):
        record = await controller.run_generation(
            include_variants=args.enable_task_generator,
        )
        logger.info(
            f"[Generation {record.generation}] Level: L{record.level} | "
            f"Tasks: {record.tasks_evaluated} | Rollouts: {record.rollouts_collected} | "
            f"Failures Indexed: {record.failures_recorded} | "
            f"Held-Out Pass Rate: {record.held_out_pass_rate * 100:.1f}% | "
            f"Mean Reward: {record.mean_reward:.3f} | "
            f"Promoted: {'YES' if record.promoted else 'NO'}"
        )

    # Save summary
    summary_path = os.path.join(args.work_dir, "self_evolution_summary.json")
    summary_data = [
        {
            "generation": r.generation,
            "level": r.level,
            "tasks_evaluated": r.tasks_evaluated,
            "rollouts_collected": r.rollouts_collected,
            "failures_recorded": r.failures_recorded,
            "held_out_pass_rate": r.held_out_pass_rate,
            "promoted": r.promoted,
            "mean_reward": r.mean_reward,
            "timestamp": r.timestamp,
        }
        for r in controller.history
    ]
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    logger.info("=" * 70)
    logger.info(f"Self-Evolution loop finished. Summary saved to {summary_path}")
    logger.info(f"Final Curriculum Level Reached: Level {controller.current_level}")
    logger.info("=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run WebRL-style online curriculum self-evolution loop for hardware.")
    parser.add_argument("--start-level", type=int, default=1, help="Starting curriculum level (1 to 7)")
    parser.add_argument("--max-level", type=int, default=7, help="Maximum curriculum level")
    parser.add_argument("--generations", type=int, default=3, help="Number of self-evolution generations to run")
    parser.add_argument("--promotion-threshold", type=float, default=0.80, help="Pass rate required to advance level")
    parser.add_argument("--enable-task-generator", action="store_true", default=True, help="Synthesize task variants")
    parser.add_argument("--work-dir", type=str, default="./sim_build/self_evolution", help="Workspace output directory")

    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
