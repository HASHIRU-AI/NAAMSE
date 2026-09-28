"""Re-score stored run prompts with a fixed referee judge, refusal-gated to save calls.

Results are appended to a JSONL file as they arrive, so an interrupted run resumes where it
stopped. Each line: run, idx, fitness score and alignment, referee score, alignment, category
verdicts, and the number of judge calls spent.

Usage:
    uv run python scripts/run_referee.py --runs 'outputs/*fit-jev*_seed*' --referee meta \\
        --out analysis-output/jev-ablation/referee_fitjev_meta.jsonl [--limit 2]
"""
import argparse
import asyncio
import glob
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
load_dotenv()  # judge API keys; must run before the judge clients are built
from src.experiments.referee import _referee_one  # noqa: E402

logger = logging.getLogger(__name__)


def load_items(pattern: str) -> List[Dict[str, Any]]:
    items = []
    for run_dir in sorted(Path(p) for p in glob.glob(pattern)):
        if not (run_dir / "final_state.json").exists():
            continue
        for idx, line in enumerate((run_dir / "prompts.jsonl").read_text().splitlines()):
            items.append({"run": run_dir.name, "idx": idx, **json.loads(line)})
    return items


def done_keys(out: Path) -> Set[Tuple[str, int]]:
    if not out.exists():
        return set()
    keys = set()
    for line in out.read_text().splitlines():
        r = json.loads(line)
        if r.get("error") is None:
            keys.add((r["run"], r["idx"]))
    return keys


async def main_async(args: argparse.Namespace) -> None:
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    items = load_items(args.runs)
    done = done_keys(out)
    todo = [it for it in items if (it["run"], it["idx"]) not in done][: args.limit or None]
    logger.info("%d prompts, %d already refereed, %d to do", len(items), len(done), len(todo))

    semaphore = asyncio.Semaphore(args.concurrency)

    async def one(it: Dict[str, Any]) -> Dict[str, Any]:
        res = await _referee_one(it, args.referee, semaphore, gated=True)
        return {"run": it["run"], "idx": it["idx"], "mutation_type": it["mutation_type"],
                "fitness_score": it["judge_score"],
                "fitness_alignment": (it.get("judge_results") or {}).get("alignment"),
                "referee": args.referee, "referee_score": res.get("score"),
                "referee_alignment": (res.get("judge_results") or {}).get("alignment"),
                "referee_categories": (res.get("judge_results") or {}).get("categories"),
                "judge_calls": res.get("judge_calls"), "error": res.get("error")}

    calls = n = 0
    with open(out, "a") as f:
        for coro in asyncio.as_completed([one(it) for it in todo]):
            rec = await coro
            f.write(json.dumps(rec) + "\n")
            f.flush()
            n += 1
            calls += rec["judge_calls"] or 0
            if n % 10 == 0 or n == len(todo):
                logger.info("%d/%d done, %d judge calls so far", n, len(todo), calls)
    logger.info("finished: %d prompts, %d judge calls (ungated would be %d)", n, calls, 7 * n)


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(message)s")
    logger.setLevel(logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", required=True, help="Glob of run directories")
    parser.add_argument("--referee", default="meta", choices=["meta", "gemini", "jev"])
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=0, help="Only referee this many (0 = all)")
    parser.add_argument("--concurrency", type=int, default=4)
    asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    main()
