"""
CLI for SPIN testing (simulation) and evaluation.

Usage (from repository root):
    python -m evaluation.cli --stage testing --study-id P_M --real-llm ...
    python -m evaluation.cli --stage evaluation --study-id P_M --skip-generation ...

Aliases: --stage 5 → testing, --stage 6 → evaluation.
"""

from __future__ import annotations

import argparse
import sys
import traceback


def _normalize_stage(token: str) -> str:
    t = str(token).strip().lower()
    if t in ("5", "testing", "test", "simulation"):
        return "testing"
    if t in ("6", "evaluation", "eval"):
        return "evaluation"
    raise ValueError(
        f"Unknown --stage {token!r}. Use testing or evaluation "
        "(aliases: 5, 6, test, simulation, eval)."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SPIN pipeline: testing (benchmark simulation) and evaluation (scoring)"
    )
    parser.add_argument(
        "--stage",
        type=str,
        required=True,
        choices=[
            "testing",
            "evaluation",
            "5",
            "6",
            "test",
            "simulation",
            "eval",
        ],
        help="testing (alias 5): run agents; evaluation (alias 6): score runs",
    )
    parser.add_argument("--study-id", type=str, help="Study id (e.g. P_M, S_T)")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument(
        "--provider",
        type=str,
        default="openai",
        choices=["openai", "openrouter"],
        help="Provider when evaluation must call an LLM (evaluator generation); OpenAI-compatible only",
    )
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--api-base", type=str, default=None)
    parser.add_argument("--real-llm", action="store_true", help="Use real LLM API (testing)")
    parser.add_argument("--n-participants", type=int, default=None)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--merge-repeats", action="store_true")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--use-cache", action="store_true")
    parser.add_argument("--cache-dir", type=str, default="results/cache")
    parser.add_argument("--profiles-json", type=str, default=None)
    parser.add_argument("--system-prompt-file", type=str, default=None)
    parser.add_argument(
        "--system-prompt-preset",
        type=str,
        default="v3_human_plus_demo",
        help="System prompt preset (testing)",
    )
    parser.add_argument("--reasoning", type=str, default="default")
    parser.add_argument("--enable-reasoning", action="store_true")
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--agent-method",
        type=str,
        default="spin",
        help="Only 'spin' is supported; 'method_01' is accepted as a legacy alias.",
    )
    parser.add_argument(
        "--spin-personality-max-tokens",
        type=int,
        default=1024,
        help="Max tokens for SPIN personality compilation",
    )
    parser.add_argument(
        "--spin-elicitation-max-tokens",
        type=int,
        default=1024,
        help="Max tokens for SPIN state elicitation",
    )
    parser.add_argument(
        "--spin-decision-max-tokens",
        type=int,
        default=1024,
        help="Max tokens for SPIN decision readout",
    )
    parser.add_argument("--spin-save-raw-prompts", action="store_true")
    parser.add_argument("--spin-save-raw-responses", action="store_true")
    parser.add_argument("--spin-no-answer-repair", action="store_true")
    parser.add_argument("--spin-repair-max-tokens", type=int, default=512)
    parser.add_argument("--spin-no-parse-retry", action="store_true")

    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="Evaluation: skip evaluator code generation",
    )
    parser.add_argument("--config-folder", type=str, default=None)

    args = parser.parse_args()
    stage = _normalize_stage(args.stage)

    agent_method = (args.agent_method or "spin").strip().lower()
    if agent_method == "method_01":
        agent_method = "spin"

    from evaluation.pipeline_core import GenerationPipeline

    pipeline_model = args.model or "gpt-4o-mini"
    pipeline = GenerationPipeline(
        provider=args.provider,
        model=pipeline_model,
        api_key=args.api_key,
        api_base=args.api_base,
    )

    try:
        if stage == "testing":
            if not args.study_id:
                raise ValueError("--study-id is required for testing")
            if agent_method != "spin":
                raise ValueError(f"Unsupported --agent-method {args.agent_method!r}; use spin")
            llm_model = args.model or "mistralai/mistral-nemo"
            agent_method_config = {
                "temperature": args.temperature,
                "max_tokens": args.spin_decision_max_tokens,
                "random_seed": args.random_seed,
                "personality_max_tokens": args.spin_personality_max_tokens,
                "elicitation_max_tokens": args.spin_elicitation_max_tokens,
                "decision_max_tokens": args.spin_decision_max_tokens,
                "save_raw_prompts": args.spin_save_raw_prompts,
                "save_raw_responses": args.spin_save_raw_responses,
                "enable_answer_repair": not args.spin_no_answer_repair,
                "answer_repair_max_tokens": args.spin_repair_max_tokens,
                "parse_error_retry": not args.spin_no_parse_retry,
            }
            result_path = pipeline.run_testing(
                args.study_id,
                use_real_llm=args.real_llm,
                model=llm_model,
                n_participants=args.n_participants,
                random_seed=args.random_seed,
                num_workers=args.num_workers,
                use_cache=args.use_cache,
                cache_dir=args.cache_dir,
                profiles_json=args.profiles_json,
                system_prompt_file=args.system_prompt_file,
                system_prompt_preset=args.system_prompt_preset,
                repeats=args.repeats,
                run_name=args.run_name,
                merge_existing_repeats=args.merge_repeats,
                reasoning=args.reasoning,
                enable_reasoning=args.enable_reasoning,
                temperature=args.temperature,
                agent_method=agent_method,
                agent_method_config=agent_method_config,
                api_base=args.api_base,
            )

        elif stage == "evaluation":
            if not args.study_id:
                raise ValueError("--study-id is required for evaluation")
            if args.skip_generation:
                print(f"Re-running evaluation for {args.study_id} (skipping evaluator generation)")
            else:
                print(f"Generating evaluator (if needed) and computing scores for {args.study_id}")
            evaluator_path = pipeline.run_evaluation(
                args.study_id,
                skip_generation=args.skip_generation,
                run_name=args.run_name,
                config_folder=args.config_folder,
            )

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
