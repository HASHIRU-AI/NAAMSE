"""Tests for the ablation runner's CLI and output writing (no network, no fuzzer run)."""
import json

import pytest

from src.experiments.ablation_config import AblationConfig
from src.experiments.run_ablation import (
    build_parser, main, make_run_dir, prompt_record, resolved_config, write_outputs,
)


def test_dry_run_prints_resolved_config(capsys):
    rc = main(["--target", "http://localhost:5000", "--action-selector", "jev", "--mutation-selector", "jev",
               "--fitness-judge", "meta", "--seed", "7", "--dry-run"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["ablation"]["action_selector"] == "jev"
    assert out["ablation"]["fitness_judge"] == "meta"
    assert out["run"]["seed"] == 7
    assert out["arm"] == "act-jev_mut-jev_fit-meta_obj-score"


def test_no_flags_is_the_baseline(capsys):
    main(["--target", "http://x", "--dry-run"])
    out = json.loads(capsys.readouterr().out)
    assert out["ablation"] == AblationConfig().to_dict()


def test_target_is_required():
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_invalid_ablation_choice_is_rejected():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--target", "http://x", "--action-selector", "nope"])


def test_run_dir_is_unique_per_arm_and_seed(tmp_path):
    a = AblationConfig()
    d1 = make_run_dir(str(tmp_path), a, 1)
    d2 = make_run_dir(str(tmp_path), a, 2)
    assert d1 != d2 and d1.is_dir() and "seed1" in d1.name and a.arm_name in d1.name


def test_write_outputs_round_trips(tmp_path):
    prompt = {
        "prompt": ["hello"], "score": 42.0,
        "conversation_history": {"messages": [{"role": "user", "content": "hello"}]},
        "metadata": {"iteration": 0, "judge_score": 42.0, "mutation_type": "echo",
                     "selector": {"action": "mutate", "action_probabilities": {"mutate": 1.0}},
                     "judge_results": {"categories": {}}},
    }
    write_outputs(tmp_path, {"all_fuzzer_prompts_with_scores": [prompt], "current_iteration": 1,
                             "covered_categories": ["violence"]})
    rows = [json.loads(line) for line in (tmp_path / "prompts.jsonl").read_text().splitlines()]
    assert rows == [prompt_record(prompt)]
    assert rows[0]["selector"]["action"] == "mutate"
    state = json.loads((tmp_path / "final_state.json").read_text())
    assert state == {"iterations_completed": 1, "n_prompts": 1, "covered_categories": ["violence"]}


def test_resolved_config_records_argv():
    args = build_parser().parse_args(["--target", "http://x", "--iterations", "3"])
    cfg = resolved_config(args, AblationConfig())
    assert cfg["run"]["iterations"] == 3 and "argv" in cfg
