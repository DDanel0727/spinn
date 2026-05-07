"""
Generation pipeline — testing (simulation) and evaluation.
"""

import json
import sys
import copy
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from datetime import datetime

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from evaluation.human_bench import HumanStudyBench
from llm.llm_participant_agent import ParticipantPool
from agents.experiment_config import get_prompt_builder, get_study_config
import evaluation.material_studies  # noqa: F401 — registers P_M / S_T configs
import time

from paths import spin_data_dir, studies_root, repo_root


class GenerationPipeline:
    """Main pipeline orchestrator"""
    
    def __init__(
        self,
        provider: str = "openai",
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ):
        """Same constructor surface as the CLI (for compatibility)."""

    def _results_method_name(self) -> Optional[str]:
        """Return the logical method bucket for results, if provided by the shell wrapper."""
        method_name = (os.getenv("RESULTS_METHOD_NAME") or "").strip()
        return method_name or None

    def _resolve_results_run_dir(self, run_name: Optional[str] = None) -> Path:
        """Resolve the base directory for testing/evaluation artifacts without changing legacy defaults."""
        method_name = self._results_method_name()
        if method_name:
            base_dir = Path("results") / method_name
            return base_dir / run_name if run_name else base_dir
        if run_name:
            return Path("results/runs") / run_name
        benchmark_folder = os.getenv("BENCHMARK_FOLDER", "benchmark")
        return Path("results") / benchmark_folder

    def _results_flat_layout(self) -> bool:
        """When RESULTS_METHOD_NAME is set (shell wrappers), store run artifacts directly under results/<method>/<run_name>/."""
        return bool(self._results_method_name())

    def _testing_result_paths(
        self,
        run_dir: Path,
        study_id: str,
        model: str,
        system_prompt_preset: str,
        reasoning: str,
        temperature: float,
    ) -> Tuple[Path, str]:
        """
        Resolve testing-run output directory and config folder label.

        Legacy: results/.../{study_id}/{model...preset}/
        Flat (RESULTS_METHOD_NAME): results/<method>/<run_name>/ (run_id already encodes study + model + time)
        """
        model_slug = model.replace("/", "_").replace("-", "_")
        prompt_slug = system_prompt_preset.replace("_", "-")
        temp_suffix = f"_temp{temperature}" if temperature != 1.0 else ""
        if reasoning and reasoning != "default" and reasoning != "low" and reasoning != "minimal":
            config_folder = f"{model_slug}_{reasoning}{temp_suffix}_{prompt_slug}"
        else:
            config_folder = f"{model_slug}{temp_suffix}_{prompt_slug}"
        if self._results_flat_layout():
            config_dir = run_dir
        else:
            config_dir = run_dir / study_id / config_folder
        return config_dir, config_folder

    
    def run_testing(
        self,
        study_id: str,
        use_real_llm: bool = False,
        model: str = "mistralai/mistral-nemo",
        n_participants: Optional[int] = None,
        random_seed: int = 42,
        num_workers: Optional[int] = None,
        use_cache: bool = False,
        cache_dir: str = "results/cache",
        profiles_json: Optional[str] = None,
        system_prompt_file: Optional[str] = None,
        system_prompt_preset: str = "v3_human_plus_demo",
        repeats: int = 1,
        run_name: Optional[str] = None,
        merge_existing_repeats: bool = False,
        reasoning: str = "default",
        enable_reasoning: bool = False,
        temperature: float = 1.0,
        agent_method: Optional[str] = None,
        agent_method_config: Optional[Dict[str, Any]] = None,
        api_base: Optional[str] = None,
    ) -> Path:
        """
        Testing: run agents and collect raw responses (benchmark / simulation).
        
        Args:
            study_id: Study ID (e.g., "study_001")
            use_real_llm: Whether to use real LLM API
            model: Model name
            n_participants: Number of participants (None = use specification)
            random_seed: Random seed
            num_workers: Number of parallel workers
            use_cache: Whether to use cache
            cache_dir: Cache directory
            profiles_json: Optional path to profiles JSON file
            system_prompt_file: Optional path to system prompt override file
            system_prompt_preset: System prompt preset
            repeats: Number of repeated runs
            run_name: Name for the run directory
            merge_existing_repeats: Whether to merge new repeats with existing results
            reasoning: Reasoning effort level
            enable_reasoning: Force enable reasoning for OpenRouter models
            
        Returns:
            Path to saved benchmark results
        """
        # Load benchmark and study
        benchmark = HumanStudyBench(str(spin_data_dir()))
        study = benchmark.load_study(study_id)
        
        # Get study config
        study_path = study.materials_path.parent
        study_config = get_study_config(study_id, study_path, study.specification)
        
        # Create prompt builder
        if hasattr(study_config, 'get_prompt_builder'):
            builder = study_config.get_prompt_builder()
        else:
            builder = get_prompt_builder(study_id)
        instructions = builder.get_instructions()
        
        # Determine n_participants and sub-study distribution
        by_sub_study_spec = study.specification.get("participants", {}).get("by_sub_study", {})
        
        # Create trials based on distribution
        if n_participants is not None:
            # When n_participants is specified, create that many trials total (1 participant per trial)
            # This ensures paper-faithful: each trial gets exactly 1 participant
            print(f"Trials: {n_participants} (one participant per trial)")
            trials = study_config.create_trials(n_trials=n_participants)
        elif by_sub_study_spec:
            # Paper-faithful: use distribution from specification
            trials = study_config.create_trials(n_trials=None)
        else:
            # Fallback: use default N
            n_def = study.specification.get('participants', {}).get('n') or 30
            print(f"Participants (default N): {n_def}")
            trials = study_config.create_trials(n_trials=n_def)
        
        _title = (study.metadata.get("title") or "").strip()
        print(
            f"Testing {study_id} · {len(trials)} trials · {_title}",
            flush=True,
        )
        
        # Load profiles if provided
        loaded_profiles = None
        if profiles_json:
            with open(profiles_json, "r", encoding="utf-8") as f:
                loaded_profiles = json.load(f)
            print(f"Loaded {len(loaded_profiles)} participant profiles from {profiles_json}")
        
        # Load system prompt override if provided
        system_prompt_override = None
        if system_prompt_file:
            with open(system_prompt_file, "r", encoding="utf-8") as f:
                system_prompt_override = f.read()
        
        # Override distribution if n_participants is manually specified (use Case 2 logic)
        if n_participants is not None:
            print(f"One-to-one: {len(trials)} trial rows")
            by_sub_study_spec = None  # Force Case 2 logic 
        
        # Cache setup
        def _slugify(text: str) -> str:
            """Slugify a string for use in folder names, preserving dots but replacing slashes/hyphens."""
            return text.replace("/", "_").replace("-", "_")
        
        model_slug = _slugify(model)
        n_tag = f"n{n_participants}" if n_participants else "auto"
        cache_path_base = Path(cache_dir) / f"{study_id}__{model_slug}__{n_tag}__{system_prompt_preset}__seed{random_seed}"
        
        # Run simulation
        start_time = time.time()
        all_runs_raw_results = []
        
        # Set up output directory structure early for incremental saving.
        run_dir = self._resolve_results_run_dir(run_name)
        config_dir, config_folder = self._testing_result_paths(
            run_dir, study_id, model, system_prompt_preset, reasoning, temperature
        )
        config_dir.mkdir(parents=True, exist_ok=True)
        incremental_output_file = config_dir / "full_benchmark.json"
        raw_responses_json = config_dir / "raw_responses.json"
        log_file_jsonl = config_dir / "raw_responses.jsonl"
        
        # SPIN participant agent (default when --agent-method omitted).
        agent_class = None
        agent_kwargs: Dict[str, Any] = {}
        am = (agent_method or "spin").strip().lower()
        if am == "method_01":
            am = "spin"
        if am != "spin":
            raise ValueError(
                f"Unknown --agent-method {agent_method!r}; this repository only supports 'spin'."
            )
        from agents import SpinParticipantAgent

        agent_class = SpinParticipantAgent
        spin_run_name = run_name or "benchmark_default"
        if self._results_method_name():
            spin_output_dir = config_dir
        else:
            spin_output_dir = Path("results/spin") / spin_run_name / study_id / config_folder
        agent_kwargs = {
            "spin_config": agent_method_config or {},
            "spin_output_dir": str(spin_output_dir),
        }

        # RESUME LOGIC: Check if we can resume from a partial run
        existing_progress_data = None
        all_runs_raw_results = []
        
        # 1. Try to load from raw_responses.json first (the most reliable full format)
        if raw_responses_json.exists():
            try:
                with open(raw_responses_json, 'r', encoding='utf-8') as f:
                    old_raw_data = json.load(f)
                
                # Check if this file already has all the data we need
                total_collected = 0
                if old_raw_data.get('all_runs_raw_responses'):
                    for run in old_raw_data['all_runs_raw_responses']:
                        total_collected += len(run.get('participants', []))
                
                # If we have data, we'll use it to resume
                if total_collected > 0:
                    print(f"Resume: {total_collected} responses in {raw_responses_json.name}")
                    # Convert raw_responses format to the internal all_runs_raw_results format
                    resumed_runs = []
                    for run in old_raw_data.get('all_runs_raw_responses', []):
                        individual_data = []
                        for p in run.get('participants', []):
                            # Construct back the individual response objects
                            for resp in p.get('raw_responses', []):
                                individual_data.append(resp)
                        resumed_runs.append({"individual_data": individual_data})
                    
                    all_runs_raw_results = resumed_runs
                    existing_progress_data = old_raw_data
            except Exception as e:
                print(f"Warning: could not read existing raw_responses.json: {e}")

        # 2. If still no data, try full_benchmark.json
        if not all_runs_raw_results and incremental_output_file.exists():
            try:
                with open(incremental_output_file, 'r', encoding='utf-8') as f:
                    existing_progress_data = json.load(f)
                if existing_progress_data.get('all_runs_raw_results'):
                    print(f"Resume: existing data in {incremental_output_file.name}")
                    all_runs_raw_results = existing_progress_data['all_runs_raw_results']
            except Exception:
                pass

        # 3. Last resort: recovery from raw_responses.jsonl
        if log_file_jsonl.exists():
            # ... (rest of log recovery logic) ...
            pass

        # PRE-FLIGHT CHECK: Is this study actually already finished?
        if existing_progress_data:
            # Calculate total participants we actually have across all repeats
            total_we_have = sum(len(run.get('individual_data', [])) for run in all_runs_raw_results)
            # Expected responses per repeat: CLI --n-participants can expand to len(trials) in 1:1 mode
            # (e.g. S_T: n_trials per sub-study → 3×n trials), so use len(trials) when set.
            # Paper-faithful by_sub_study also builds len(trials) one-to-one rows; participants.n alone
            # (e.g. 80) is not the API call count (can be 200), so never use only that here.
            if n_participants is not None:
                expected_per_repeat = len(trials)
            elif by_sub_study_spec:
                expected_per_repeat = len(trials)
            else:
                expected_n = study.specification.get("participants", {}).get("n", 30)
                expected_per_repeat = expected_n
            expected_total = expected_per_repeat * repeats
            if total_we_have >= expected_total:
                print(f"Skip simulation: {study_id} already complete ({total_we_have} responses).")
                return incremental_output_file

        # Create initial file ONLY if not resuming
        if not existing_progress_data:
            from datetime import datetime as _datetime_module
            try:
                initial_data = {
                    "timestamp": _datetime_module.now().strftime("%Y%m%d_%H%M%S"),
                    "study_id": study_id,
                    "title": study.metadata.get('title', ''),
                    "model": model,
                    "reasoning": reasoning,
                    "use_real_llm": use_real_llm,
                    "system_prompt_preset": system_prompt_preset,
                    "random_seed": random_seed,
                    "repeats_completed": 0,
                    "repeats_total": repeats,
                    "status": "starting",
                    "all_runs_raw_results": []
                }
                with open(incremental_output_file, 'w', encoding='utf-8', errors='replace') as f:
                    json.dump(initial_data, f, indent=2, ensure_ascii=False)
                print(f"Checkpoint: {incremental_output_file}", flush=True)
            except Exception as e:
                print(f"Warning: could not create initial save file: {e}", flush=True)
        
        for r_idx in range(repeats):
            if repeats > 1:
                print(f"\n>>> Run {r_idx + 1}/{repeats}")
            
            r_tag = f"_r{r_idx}" if repeats > 1 else ""
            cache_path = Path(f"{cache_path_base}{r_tag}.json")
            
            current_run_raw_results = None
            
            # RESUME: skip finished repeats; continue partial repeats with existing rows
            existing_repeat_responses = None
            if r_idx < len(all_runs_raw_results):
                prev_repeat = all_runs_raw_results[r_idx].get('individual_data', []) or []
                if len(prev_repeat) >= len(trials):
                    print(f"Repeat {r_idx + 1}/{repeats} already complete ({len(prev_repeat)} responses), skip.")
                    continue
                existing_repeat_responses = prev_repeat
                print(f"Resume repeat {r_idx + 1}/{repeats} ({len(existing_repeat_responses)} responses so far)")

            if use_cache and cache_path.exists():
                print(f"Load cache: {cache_path}")
                try:
                    with open(cache_path, 'r', encoding='utf-8', errors='replace') as f:
                        cached = json.load(f)
                    current_run_raw_results = cached.get('raw_results') or cached
                except (UnicodeDecodeError, json.JSONDecodeError) as e:
                    print(f"Warning: cache encoding/JSON error at position {getattr(e, 'start', '?')}, will regenerate: {e}")
                    # Delete corrupted cache file
                    try:
                        cache_path.unlink()
                        print("  Deleted corrupted cache file")
                    except:
                        pass
                except Exception as e:
                    print(f"Warning: cache file error: {e}, will regenerate")
            
            if current_run_raw_results is None:
                # Check if study requires group trials (multi-participant, sequential rounds)
                requires_group_trials = getattr(study_config, 'REQUIRES_GROUP_TRIALS', False)
                
                if requires_group_trials:
                    # Special handling for studies requiring group interaction.
                    print(f"Using group experiment runner for {study_id}")
                    
                    # Prepare participant pool kwargs
                    participant_pool_kwargs = {
                        "study_specification": study.specification,
                        "use_real_llm": use_real_llm,
                        "model": model,
                        "random_seed": random_seed + r_idx,
                        "num_workers": num_workers or 1,
                        "profiles": loaded_profiles,
                        "prompt_builder": builder,
                        "system_prompt_override": system_prompt_override,
                        "system_prompt_preset": system_prompt_preset,
                        "reasoning": reasoning,
                        "enable_reasoning": enable_reasoning,
                        "temperature": temperature,
                        "api_base": api_base,
                        "agent_class": agent_class,
                        "agent_kwargs": agent_kwargs,
                    }
                    
                    # Call custom group experiment runner
                    current_run_raw_results = study_config.run_group_experiment(
                        trials=trials,
                        instructions=instructions,
                        participant_pool_kwargs=participant_pool_kwargs,
                        prompt_builder=builder
                    )
                
                # Standard experiment flow: 1 participant per trial (one-to-one mapping)
                elif n_participants is not None or by_sub_study_spec:
                    # Use one-to-one mode: create a single pool with n_participants = len(trials)
                    # This is more efficient than creating multiple pools (avoids thread contention)
                    # Ensure n_participants matches len(trials) for one_to_one mode
                    actual_n_participants = n_participants if n_participants is not None else len(trials)
                    
                    if actual_n_participants != len(trials):
                        print(f"Warning: n_participants ({actual_n_participants}) != len(trials) ({len(trials)}); using {len(trials)} for one-to-one mode.")
                        actual_n_participants = len(trials)
                    
                    
                    # Create a single pool with all participants
                    pool = ParticipantPool(
                        study_specification=study.specification,
                        n_participants=actual_n_participants,
                        use_real_llm=use_real_llm,
                        model=model,
                        random_seed=random_seed + r_idx,
                        num_workers=num_workers,  # Use full parallelism here
                        profiles=loaded_profiles[:actual_n_participants] if loaded_profiles else None,
                        prompt_builder=builder,
                        system_prompt_override=system_prompt_override,
                        system_prompt_preset=system_prompt_preset,
                        study_id=study_id,
                        reasoning=reasoning,
                        enable_reasoning=enable_reasoning,
                        existing_responses=existing_repeat_responses,
                        temperature=temperature,
                        api_base=api_base,
                        agent_class=agent_class,
                        agent_kwargs=agent_kwargs,
                    )
                    
                    # Create save callback that saves after each API call returns
                    # Use throttling to avoid saving too frequently (max once per second)
                    # Import datetime at function level to avoid scoping issues in closures
                    from datetime import datetime as _dt_module
                    last_save_time = [0]  # Use list to allow modification in closure
                    last_progress_print = [0]  # Track last progress print time
                    def save_after_api_call(new_resp_data=None):
                        """Save current state after each API call completes"""
                        import time as _time
                        current_time = _time.time()
                        
                        try:
                            # 1. PROGRESSIVE LOGGING: Append to JSONL immediately (No throttle!)
                            if new_resp_data:
                                try:
                                    with open(log_file_jsonl, 'a', encoding='utf-8') as f:
                                        f.write(json.dumps(new_resp_data, ensure_ascii=False) + "\n")
                                except Exception as e:
                                    pass # Don't let log failure stop the run

                            # 2. TERMINAL UPDATE: Show progress immediately
                            # Get current count from pool
                            current_progress = sum(len(p.trial_responses) for p in pool.participants)
                            total_trials = len(trials)
                            
                            # Print progress immediately whenever there's a change
                            progress_pct = (current_progress / total_trials * 100) if total_trials > 0 else 0
                            print(f"\r   progress {current_progress}/{total_trials} trials ({progress_pct:.1f}%)  repeat {r_idx + 1}/{repeats}", end='', flush=True)
                            
                            # 3. THROTTLED DISK SAVE: Update full_benchmark.json every 5 seconds
                            if current_time - last_save_time[0] < 5.0:
                                return
                            last_save_time[0] = current_time
                            
                            # Get current partial results from pool for the main file
                            partial_results = pool.aggregate_results()
                            
                            # In-progress save: only completed repeats before r_idx, then current partial
                            # (avoid all_runs_raw_results + [partial] which duplicates the active repeat on disk)
                            base_slice = [
                                {"individual_data": list(r.get("individual_data", []) or [])}
                                for r in all_runs_raw_results[:r_idx]
                            ]
                            partial_idata = partial_results.get("individual_data", []) or []
                            incremental_runs = base_slice + [{"individual_data": partial_idata}]
                            incremental_data = {
                                "timestamp": _dt_module.now().strftime("%Y%m%d_%H%M%S"),
                                "study_id": study_id,
                                "title": study.metadata.get('title', ''),
                                "model": model,
                                "reasoning": reasoning,
                                "use_real_llm": use_real_llm,
                                "system_prompt_preset": system_prompt_preset,
                                "random_seed": random_seed,
                                "repeats_completed": r_idx,
                                "repeats_total": repeats,
                                "current_repeat_progress": current_progress,
                                "status": "in_progress",
                                "all_runs_raw_results": incremental_runs
                            }
                            # Atomic save: write to tmp then rename
                            tmp_file = incremental_output_file.with_suffix('.tmp')
                            with open(tmp_file, 'w', encoding='utf-8', errors='replace') as f:
                                json.dump(incremental_data, f, indent=2, ensure_ascii=False)
                            tmp_file.replace(incremental_output_file)
                        except Exception as e:
                            pass
                    
                    # Run experiment in one-to-one mode (each participant runs exactly one trial)
                    # This uses ParticipantPool's internal ThreadPoolExecutor with num_workers
                    # This is much faster than creating multiple pools with num_workers=1
                    print(
                        f"\nSimulate · SPIN · trials={len(trials)} participants={actual_n_participants} workers={num_workers or 1} mode=one-to-one",
                        flush=True,
                    )
                    
                    # Add a progress monitor thread to show updates more frequently
                    import threading
                    progress_stop = threading.Event()
                    def progress_monitor():
                        """Monitor and print progress every 0.5 seconds"""
                        last_count = 0
                        while not progress_stop.is_set():
                            try:
                                current_count = sum(len(p.trial_responses) for p in pool.participants)
                                if current_count > last_count:
                                    progress_pct = (current_count / len(trials) * 100) if len(trials) > 0 else 0
                                    print(f"\r   progress {current_count}/{len(trials)} trials ({progress_pct:.1f}%)  repeat {r_idx + 1}/{repeats}", end='', flush=True)
                                    last_count = current_count
                            except:
                                pass
                            progress_stop.wait(0.5)  # Check every 0.5 seconds
                    
                    monitor_thread = threading.Thread(target=progress_monitor, daemon=True)
                    monitor_thread.start()
                    
                    try:
                        current_run_raw_results = pool.run_experiment(
                            trials, 
                            instructions, 
                            prompt_builder=builder,
                            one_to_one=True,  # Enable one-to-one mode for better performance
                            save_callback=save_after_api_call  # Save after each API call
                        )
                    finally:
                        progress_stop.set()  # Stop the monitor
                
                # Case 3: Fallback default
                else:
                    n_def = study.specification.get('participants', {}).get('n') or 30
                    print(f"Participants (default N): {n_def}")
                    pool = ParticipantPool(
                        study_specification=study.specification,
                        n_participants=n_def,
                        use_real_llm=use_real_llm,
                        model=model,
                        random_seed=random_seed + r_idx,
                        num_workers=num_workers,
                        profiles=loaded_profiles,
                        prompt_builder=builder,
                        system_prompt_override=system_prompt_override,
                        system_prompt_preset=system_prompt_preset,
                        study_id=study_id,
                        reasoning=reasoning,
                        enable_reasoning=enable_reasoning,
                        existing_responses=existing_repeat_responses,
                        temperature=temperature,
                        api_base=api_base,
                        agent_class=agent_class,
                        agent_kwargs=agent_kwargs,
                    )
                    
                    # Create save callback that saves after each API call returns
                    # Use throttling to avoid saving too frequently (max once per second)
                    # Import datetime at function level to avoid scoping issues in closures
                    from datetime import datetime as _dt_module_fallback
                    last_save_time_fallback = [0]  # Use list to allow modification in closure
                    last_progress_print_fallback = [0]  # Track last progress print time
                    def save_after_api_call_fallback(new_resp_data=None):
                        """Save current state after each API call completes"""
                        import time as _time
                        current_time = _time.time()
                        
                        try:
                            # 1. PROGRESSIVE LOGGING: Append to JSONL immediately
                            if new_resp_data:
                                try:
                                    with open(log_file_jsonl, 'a', encoding='utf-8') as f:
                                        f.write(json.dumps(new_resp_data, ensure_ascii=False) + "\n")
                                except Exception:
                                    pass

                            # 2. TERMINAL UPDATE: Show progress immediately
                            current_progress = sum(len(p.trial_responses) for p in pool.participants)
                            total_expected = n_def * len(trials) if n_def else len(trials) * len(pool.participants)
                            
                            # Print progress immediately
                            progress_pct = (current_progress / total_expected * 100) if total_expected > 0 else 0
                            print(f"\r   progress {current_progress}/{total_expected} responses ({progress_pct:.1f}%)  repeat {r_idx + 1}/{repeats}", end='', flush=True)
                            
                            # 3. THROTTLED DISK SAVE: Update full_benchmark.json every 5 seconds
                            if current_time - last_save_time_fallback[0] < 5.0:
                                return
                            last_save_time_fallback[0] = current_time
                            
                            # Get current partial results from pool
                            partial_results = pool.aggregate_results()
                            
                            base_slice = [
                                {"individual_data": list(r.get("individual_data", []) or [])}
                                for r in all_runs_raw_results[:r_idx]
                            ]
                            partial_idata = partial_results.get("individual_data", []) or []
                            incremental_runs = base_slice + [{"individual_data": partial_idata}]
                            incremental_data = {
                                "timestamp": _dt_module_fallback.now().strftime("%Y%m%d_%H%M%S"),
                                "study_id": study_id,
                                "title": study.metadata.get('title', ''),
                                "model": model,
                                "reasoning": reasoning,
                                "use_real_llm": use_real_llm,
                                "system_prompt_preset": system_prompt_preset,
                                "random_seed": random_seed,
                                "repeats_completed": r_idx,
                                "repeats_total": repeats,
                                "current_repeat_progress": current_progress,
                                "status": "in_progress",
                                "all_runs_raw_results": incremental_runs
                            }
                            # Atomic save: write to tmp then rename
                            tmp_file = incremental_output_file.with_suffix('.tmp')
                            with open(tmp_file, 'w', encoding='utf-8', errors='replace') as f:
                                json.dump(incremental_data, f, indent=2, ensure_ascii=False)
                            tmp_file.replace(incremental_output_file)
                        except Exception as e:
                            pass
                    
                    print(
                        f"\nSimulate · SPIN · trials_per_ppt={len(trials)} participants={n_def} workers={num_workers or 1}",
                        flush=True,
                    )
                    
                    # Add a progress monitor thread to show updates more frequently
                    import threading
                    progress_stop_fallback = threading.Event()
                    def progress_monitor_fallback():
                        """Monitor and print progress every 0.5 seconds"""
                        last_count = 0
                        total_expected = n_def * len(trials)
                        while not progress_stop_fallback.is_set():
                            try:
                                current_count = sum(len(p.trial_responses) for p in pool.participants)
                                if current_count > last_count:
                                    progress_pct = (current_count / total_expected * 100) if total_expected > 0 else 0
                                    print(f"\r   progress {current_count}/{total_expected} responses ({progress_pct:.1f}%)  repeat {r_idx + 1}/{repeats}", end='', flush=True)
                                    last_count = current_count
                            except:
                                pass
                            progress_stop_fallback.wait(0.5)  # Check every 0.5 seconds
                    
                    monitor_thread_fallback = threading.Thread(target=progress_monitor_fallback, daemon=True)
                    monitor_thread_fallback.start()
                    
                    try:
                        current_run_raw_results = pool.run_experiment(
                            trials, 
                            instructions, 
                            prompt_builder=builder,
                            save_callback=save_after_api_call_fallback  # Save after each API call
                        )
                    finally:
                        progress_stop_fallback.set()  # Stop the monitor
                
                # Save cache
                if use_cache:
                    payload = {
                        "version": 1,
                        "study_id": study_id,
                        "repeat_idx": r_idx,
                        "raw_results": current_run_raw_results,
                    }
                    with open(cache_path, 'w', encoding='utf-8', errors='replace') as f:
                        json.dump(payload, f, ensure_ascii=False)
            
            # Print completion message
            num_responses = len(current_run_raw_results.get('individual_data', []))
            print(f"Repeat {r_idx + 1}/{repeats} done: {num_responses} responses", flush=True)
            
            if r_idx < len(all_runs_raw_results):
                all_runs_raw_results[r_idx] = current_run_raw_results
            else:
                all_runs_raw_results.append(current_run_raw_results)
            
            # INCREMENTAL SAVE: Save results after each repeat to prevent data loss
            # (Output directory structure is set up before the loop)
            try:
                # Prepare incremental save data
                # Use function-level datetime import to avoid scoping issues
                from datetime import datetime as _dt_module_incremental
                incremental_runs = [{"individual_data": run_data.get('individual_data', [])} for run_data in all_runs_raw_results]
                incremental_data = {
                    "timestamp": _dt_module_incremental.now().strftime("%Y%m%d_%H%M%S"),
                    "study_id": study_id,
                    "title": study.metadata.get('title', ''),
                    "model": model,
                    "reasoning": reasoning,
                    "use_real_llm": use_real_llm,
                    "system_prompt_preset": system_prompt_preset,
                    "random_seed": random_seed,
                    "repeats_completed": len(all_runs_raw_results),
                    "repeats_total": repeats,
                    "status": "in_progress" if len(all_runs_raw_results) < repeats else "complete",
                    "all_runs_raw_results": incremental_runs
                }
                
                # Save incrementally (will be overwritten at final save)
                with open(incremental_output_file, 'w', encoding='utf-8', errors='replace') as f:
                    json.dump(incremental_data, f, indent=2, ensure_ascii=False)
                
                print(f"Saved progress: {len(all_runs_raw_results)}/{repeats} repeats", flush=True)
            except Exception as e:
                print(f"Warning: could not save incrementally: {e}", flush=True)
        
        # Try to aggregate results (optional)
        try:
            # Use the first run's data for aggregation stats
            aggregation_source = all_runs_raw_results[0] if all_runs_raw_results else {"individual_data": []}
            results = study_config.aggregate_results(aggregation_source)
        except Exception as e:
            print(f"Warning: Failed to aggregate results: {e}")
            results = {"descriptive_statistics": {}, "inferential_statistics": {}, "error": str(e)}
        
        # Save results - simplified structure
        # Use global datetime import from top of file
        run_dir = self._resolve_results_run_dir(run_name)
        config_dir, _config_folder_tag = self._testing_result_paths(
            run_dir, study_id, model, system_prompt_preset, reasoning, temperature
        )
        config_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = config_dir / "full_benchmark.json"
        
        # Load existing results if merging repeats
        existing_runs = []
        existing_metadata = {}
        if merge_existing_repeats and output_file.exists():
            try:
                with open(output_file, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                
                # Extract existing runs
                if existing_data.get('all_runs_raw_results'):
                    existing_runs = existing_data['all_runs_raw_results']
                elif existing_data.get('individual_data'):
                    # Convert single run to list format
                    existing_runs = [{"individual_data": existing_data['individual_data']}]
                
                # Preserve metadata from existing file
                existing_metadata = {
                    "timestamp": existing_data.get('timestamp', timestamp),
                    "title": existing_data.get('title', study.metadata['title']),
                    "model": existing_data.get('model', model),
                    "use_real_llm": existing_data.get('use_real_llm', use_real_llm),
                    "system_prompt_preset": existing_data.get('system_prompt_preset', system_prompt_preset),
                    "random_seed": existing_data.get('random_seed', random_seed),
                }
                
                print(f"Merging: {len(existing_runs)} existing run(s) + {repeats} new")
            except Exception as e:
                print(f"Warning: could not load existing results for merging: {e}")
                existing_runs = []
        
        # Merge new runs with existing runs
        all_merged_runs = existing_runs + [
            {"individual_data": run_data.get('individual_data', [])}
            for run_data in all_runs_raw_results
        ]
        
        total_repeats = len(all_merged_runs)
        
        # Helper function to remove raw_response_text from response data (keep full_benchmark.json small)
        def clean_response_data(data):
            """Remove raw_response_text from response objects and sanitize text fields for UTF-8"""
            if isinstance(data, dict):
                cleaned = {k: v for k, v in data.items() if k != 'raw_response_text'}
                for k, v in cleaned.items():
                    cleaned[k] = clean_response_data(v)
                return cleaned
            elif isinstance(data, list):
                return [clean_response_data(item) for item in data]
            elif isinstance(data, str):
                # Sanitize string values to ensure UTF-8 compatibility
                try:
                    # Try to encode as UTF-8 to check validity
                    data.encode('utf-8')
                    return data
                except (UnicodeEncodeError, UnicodeDecodeError):
                    # Replace invalid characters
                    return data.encode('utf-8', errors='replace').decode('utf-8')
            else:
                return data
        
        # Extract raw responses before cleaning
        raw_responses_data = {
            "timestamp": existing_metadata.get("timestamp", timestamp),
            "study_id": study_id,
            "title": existing_metadata.get("title", study.metadata['title']),
            "model": existing_metadata.get("model", model),
            "reasoning": existing_metadata.get("reasoning", reasoning),
            "use_real_llm": existing_metadata.get("use_real_llm", use_real_llm),
            "system_prompt_preset": existing_metadata.get("system_prompt_preset", system_prompt_preset),
            "random_seed": existing_metadata.get("random_seed", random_seed),
            "repeats": total_repeats,
            "all_runs_raw_responses": []
        }
        
        # Extract raw responses from all runs
        for run_idx, run_data in enumerate(all_merged_runs):
            participants_data = run_data.get('individual_data', [])
            run_raw_responses = []
            
            # Check if data is flat structure (legacy format) or nested structure (from testing run)
            # Flat structure: individual_data is a list of trial responses (each has participant_id, response_text, etc.)
            # Nested structure: individual_data is a list of participants (each has participant_id, responses[])
            # Note: Flat structure is kept for backward compatibility with legacy results
            is_flat_structure = len(participants_data) > 0 and 'responses' not in participants_data[0]
            
            if is_flat_structure:
                # Handle flat structure: group responses by participant_id
                participant_responses_map = defaultdict(list)
                
                for response_item in participants_data:
                    participant_id = response_item.get('participant_id', 0)
                    # Make a deep copy to avoid modifying the original
                    raw_resp = copy.deepcopy(response_item)
                    
                    # Remove items field from trial_info to reduce file size
                    if "trial_info" in raw_resp and isinstance(raw_resp["trial_info"], dict):
                        if "items" in raw_resp["trial_info"]:
                            del raw_resp["trial_info"]["items"]
                    
                    participant_responses_map[participant_id].append(raw_resp)
                
                # Convert to nested format for raw_responses.json
                for participant_id, responses in participant_responses_map.items():
                    participant_raw_responses = {
                        "participant_id": participant_id,
                        "raw_responses": responses
                    }
                    run_raw_responses.append(participant_raw_responses)
            else:
                # Handle nested structure (original testing-run format)
                for participant_data in participants_data:
                    participant_id = participant_data.get('participant_id', 0)
                    responses = participant_data.get('responses', [])
                    participant_raw_responses = {
                        "participant_id": participant_id,
                        "raw_responses": []
                    }
                    
                    for response in responses:
                        # Save complete response dictionary to prevent information loss
                        # Make a deep copy to avoid modifying the original
                        raw_resp = copy.deepcopy(response)
                        
                        # Remove items field from trial_info to reduce file size
                        if "trial_info" in raw_resp and isinstance(raw_resp["trial_info"], dict):
                            if "items" in raw_resp["trial_info"]:
                                del raw_resp["trial_info"]["items"]
                        
                        participant_raw_responses["raw_responses"].append(raw_resp)
                    
                    run_raw_responses.append(participant_raw_responses)
            
            raw_responses_data["all_runs_raw_responses"].append({
                "run_index": run_idx,
                "participants": run_raw_responses
            })
        
        # Prepare study data (cleaned, without raw_response_text)
        # Use all_merged_runs[0] as the primary data source for summary stats
        primary_run_data = all_merged_runs[0] if all_merged_runs else {"individual_data": []}
        
        cleaned_individual_data = clean_response_data(primary_run_data.get('individual_data', []))
        cleaned_all_runs = [{"individual_data": clean_response_data(run_data.get('individual_data', []))} for run_data in all_merged_runs] if total_repeats > 1 else None
        
        # Calculate overall usage statistics for the saved data
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_tokens = 0
        total_cost = 0.0
        total_participants_all_runs = 0
        
        for run in all_merged_runs:
            participants_data = run.get('individual_data', [])
            is_flat = len(participants_data) > 0 and 'responses' not in participants_data[0]
            
            if is_flat:
                # Handle flat structure
                total_participants_all_runs += len(set(resp.get('participant_id') for resp in participants_data))
                for resp in participants_data:
                    usage = resp.get('usage', {})
                    total_prompt_tokens += usage.get('prompt_tokens', 0) or 0
                    total_completion_tokens += usage.get('completion_tokens', 0) or 0
                    total_tokens += usage.get('total_tokens', 0) or 0
                    total_cost += usage.get('cost', 0.0) or 0.0
            else:
                # Handle nested structure
                total_participants_all_runs += len(participants_data)
                for participant in participants_data:
                    for resp in participant.get('responses', []):
                        usage = resp.get('usage', {})
                        total_prompt_tokens += usage.get('prompt_tokens', 0) or 0
                        total_completion_tokens += usage.get('completion_tokens', 0) or 0
                        total_tokens += usage.get('total_tokens', 0) or 0
                        total_cost += usage.get('cost', 0.0) or 0.0
        
        avg_tokens_per_participant = total_tokens / total_participants_all_runs if total_participants_all_runs > 0 else 0
        avg_cost_per_participant = total_cost / total_participants_all_runs if total_participants_all_runs > 0 else 0

        save_data = {
            "timestamp": existing_metadata.get("timestamp", timestamp),
            "study_id": study_id,
            "title": existing_metadata.get("title", study.metadata['title']),
            "model": existing_metadata.get("model", model),
            "reasoning": existing_metadata.get("reasoning", reasoning),
            "use_real_llm": existing_metadata.get("use_real_llm", use_real_llm),
            "system_prompt_preset": existing_metadata.get("system_prompt_preset", system_prompt_preset),
            "random_seed": existing_metadata.get("random_seed", random_seed),
            "elapsed_time": time.time() - start_time,
            "repeats": total_repeats,
            "usage_stats": {
                "total_prompt_tokens": total_prompt_tokens,
                "total_completion_tokens": total_completion_tokens,
                "total_tokens": total_tokens,
                "total_cost": float(total_cost),
                "avg_tokens_per_participant": float(avg_tokens_per_participant),
                "avg_cost_per_participant": float(avg_cost_per_participant)
            },
            "descriptive_statistics": results.get('descriptive_statistics', {}),
            "inferential_statistics": results.get('inferential_statistics', {}),
            "individual_data": cleaned_individual_data,
            "all_runs_raw_results": cleaned_all_runs,
            "summary": {
                "total_participants": len(cleaned_individual_data) if cleaned_individual_data else 0
            }
        }
        
        # Save (overwrite with merged data)
        with open(output_file, 'w', encoding='utf-8', errors='replace') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        
        # Save raw responses to separate file
        raw_responses_file = config_dir / "raw_responses.json"
        with open(raw_responses_file, 'w', encoding='utf-8', errors='replace') as f:
            json.dump(raw_responses_data, f, indent=2, ensure_ascii=False)
        
        elapsed = time.time() - start_time
        print(f"\nTesting complete · {elapsed:.1f}s")
        print(f"  results: {output_file}")
        print(f"  raw:     {raw_responses_file}")
        if merge_existing_repeats and total_repeats > repeats:
            print(f"  - Participants: {n_participants}, Total Runs: {total_repeats} ({len(existing_runs)} existing + {repeats} new)")
        else:
            print(f"  - Participants: {n_participants}, Runs: {total_repeats}")
        print(f"\nNext step: run evaluation to generate scores")
        print(f"  python -m evaluation.cli --stage evaluation --study-id {study_id}")
        
        return output_file
    
    def run_evaluation(
        self,
        study_id: str,
        study_dir: Optional[Path] = None,
        skip_generation: bool = False,
        run_name: Optional[str] = None,
        config_folder: Optional[str] = None,
        disable_formatter: bool = True,
    ) -> Path:
        """
        Evaluation: score benchmark outputs with alignment metrics (P_M / S_T).
        
        Args:
            study_id: Study ID (e.g., "study_001")
            study_dir: Study directory path
            skip_generation: Skip evaluator code generation
            run_name: Name of the run
            config_folder: Specific config folder to evaluate (optional)
            
        Returns:
            Path to generated evaluator file
        """
        print(f"Running evaluation (scoring) for {study_id}")
        import json
        import os as _os_module
        from pathlib import Path
        
        # Determine study directory
        if study_dir is None:
            study_dir = studies_root() / study_id
        
        study_dir = Path(study_dir)
        if not study_dir.exists():
            raise FileNotFoundError(f"Study directory not found: {study_dir}")
        
        # Check required files exist
        required_files = ["ground_truth.json", "specification.json"]
        for fname in required_files:
            if not (study_dir / fname).exists():
                raise FileNotFoundError(f"Required file not found: {study_dir / fname}")
        
        from evaluation import material_studies

        # Generate evaluator (unless skipping)
        if not skip_generation:
            raise RuntimeError(
                "This repository does not generate evaluator code; use --skip-generation."
            )
        else:
            if material_studies.is_bundled_evaluator(study_id):
                evaluator_path = material_studies.EVALUATOR_MODULE_PATH
                print(f"  - Sanity check / bundled reference: {evaluator_path}")
            else:
                evaluator_path = repo_root() / "evaluators" / f"{study_id}_evaluator.py"
                if not evaluator_path.exists():
                    raise FileNotFoundError(
                        f"Evaluator not found: {evaluator_path}. "
                        "Add the file to the repo or supply your own; auto-generation is not supported here."
                    )
                print(f"  - Using existing evaluator: {evaluator_path}")
        
        # Find results base directory (must match testing run / shell wrappers)
        results_base_dir = self._resolve_results_run_dir(run_name)
        
        if self._results_flat_layout():
            results_study_dir = results_base_dir
            if not results_study_dir.exists():
                raise FileNotFoundError(f"Run results directory not found: {results_study_dir}. Run testing first.")
            if (results_study_dir / "full_benchmark.json").exists():
                all_config_dirs = [results_study_dir]
            else:
                all_config_dirs = [d for d in results_study_dir.iterdir() if d.is_dir()]
            if not all_config_dirs:
                raise FileNotFoundError(f"No benchmark results found in {results_study_dir}. Run testing first.")
        else:
            results_study_dir = results_base_dir / study_id
            if not results_study_dir.exists():
                raise FileNotFoundError(f"Study results directory not found: {results_study_dir}. Run testing first.")
            all_config_dirs = [d for d in results_study_dir.iterdir() if d.is_dir()]
            if not all_config_dirs:
                raise FileNotFoundError(f"No config folders found in {results_study_dir}")
        
        if config_folder:
            if self._results_flat_layout():
                print(
                    "  - Note: --config-folder is ignored for flat runs (results under results/<method>/<run_name>/).",
                    flush=True,
                )
                config_dirs_to_process = sorted(all_config_dirs)
            else:
                config_dirs_to_process = [d for d in all_config_dirs if d.name == config_folder]
                if not config_dirs_to_process:
                    raise FileNotFoundError(f"Specific config folder not found: {config_folder}")
        else:
            config_dirs_to_process = sorted(all_config_dirs)
            if len(config_dirs_to_process) > 1:
                print(f"  - Found {len(config_dirs_to_process)} config folders, evaluating all...")
        
        last_evaluator_path = evaluator_path

        print(f"  Processing {len(config_dirs_to_process)} config folder(s)...")
        for idx, cfg_dir in enumerate(config_dirs_to_process, 1):
            print(f"\n  >>> [{idx}/{len(config_dirs_to_process)}] Evaluating: {cfg_dir.name}")
            
            benchmark_file = cfg_dir / "full_benchmark.json"
            if not benchmark_file.exists():
                print(f"  Warning: benchmark file not found in {cfg_dir.name}, skipping.")
                continue
            
            # ===== Load benchmark data =====
            print(f"  - Loading benchmark data...")
            with open(benchmark_file, 'r', encoding='utf-8') as f:
                benchmark_data = json.load(f)            
            print(f"    Loaded benchmark data")
            
            # ===== Calculate Raw Failure Rate =====
            print(f"  - Calculating raw failure rate...")
            from evaluation.sanity_check import calculate_raw_failure_rate
            
            raw_failure_stats = calculate_raw_failure_rate(benchmark_data)
            raw_failure_rate = raw_failure_stats.get("raw_failure_rate", 0.0)
            print(f"    Raw failure rate: {raw_failure_rate:.2f}% ({raw_failure_stats.get('raw_failed', 0)}/{raw_failure_stats.get('raw_total', 0)})")
            if raw_failure_stats.get('raw_failure_breakdown'):
                breakdown = raw_failure_stats['raw_failure_breakdown']
                print(f"      Breakdown: {breakdown.get('empty', 0)} empty, {breakdown.get('refusal', 0)} refusal, {breakdown.get('other', 0)} other")
            
            # ===== Sanity Check: Verify response extraction (Final Failure Rate) =====
            print(f"  - Running sanity check for response extraction...")
            from evaluation.sanity_check import run_sanity_check, format_failed_responses
            
            # Count total responses for progress indication
            total_resp_count = len(benchmark_data.get('individual_data', []))
            if total_resp_count > 0:
                print(f"    Checking {total_resp_count} responses...")
            
            sanity_check_result = run_sanity_check(study_id, benchmark_file, evaluator_path)
            
            if not sanity_check_result.get("all_passed", True):
                failed_responses = sanity_check_result.get("failed_responses", [])
                print(f"  Warning: {len(failed_responses)}/{sanity_check_result.get('total_checked', 0)} responses cannot be fully extracted")
                
                # Print one failed example for inspection
                if failed_responses:
                    example = failed_responses[0]
                    print(f"\n  Example failed response:")
                    print(f"    Participant {example['participant_id']}, Response {example['response_index']}")
                    print(f"    Missing Q numbers: {example['missing_q_numbers']}")
                    print(f"    Required: {example['required_q_numbers']}")
                    print(f"    Extracted: {example['extracted_q_numbers']}")
                    print(f"    Response preview: {example['response_text_preview'][:300]}")
                
                # Formatter disabled - skip automatic formatting of failed responses
                # disable_formatter=True means formatter is disabled (default)
                # Only run formatter if disable_formatter is False (explicitly enabled)
                if not disable_formatter:
                    print(f"  - Activating formatter for failed responses (multithreading enabled)...")
                    
                    # Format failed responses in parallel
                    formatted_count = format_failed_responses(
                        study_id=study_id,
                        benchmark_file=benchmark_file,
                        failed_responses=failed_responses,
                        evaluator_path=evaluator_path,
                        num_workers=32
                    )
                    
                    print(f"  Formatted {formatted_count}/{len(failed_responses)} responses")
                    
                    # Re-run sanity check after formatting
                    sanity_check_result = run_sanity_check(study_id, benchmark_file, evaluator_path)
                    if not sanity_check_result.get("all_passed", True):
                        remaining_failed = len(sanity_check_result.get("failed_responses", []))
                        print(f"  Warning: {remaining_failed} responses still cannot be extracted after formatting")
                    else:
                        print(f"  All responses passed sanity check after formatting")
                else:
                    print(f"  Warning: skipping formatter (disabled); {len(failed_responses)} responses failed extraction")
            else:
                total_checked = sanity_check_result.get('total_checked', 0)
                passed = sanity_check_result.get('passed', 0)
                skipped = sanity_check_result.get('skipped_responses', 0)
                total = sanity_check_result.get('total_responses', total_checked)
                if skipped > 0:
                    print(f"  All responses passed sanity check ({passed}/{total_checked} checked, {skipped} skipped out of {total} total)")
                else:
                    print(f"  All responses passed sanity check ({passed}/{total_checked})")
            
            # ===== Use already loaded benchmark data for evaluation =====
            individual_data = benchmark_data.get('individual_data', [])
            all_runs_data = []
            if benchmark_data.get('all_runs_raw_results'):
                for run_data in benchmark_data['all_runs_raw_results']:
                    all_runs_data.append(run_data.get('individual_data', []))
            else:
                all_runs_data = [individual_data] if individual_data else []
            
            if not all_runs_data or not all_runs_data[0]:
                print(f"  Warning: no raw response data in benchmark for {cfg_dir.name}")
                continue
            
            # Combine all participant data from all runs into a single pool
            combined_participant_pool = []
            for run_individual_data in all_runs_data:
                if run_individual_data:
                    combined_participant_pool.extend(run_individual_data)
            
            if not combined_participant_pool:
                print(f"    No participant data available for bootstrap")
                continue
            
            # Check if data is flat structure (legacy format) or nested structure (from testing run)
            # Note: Flat structure is kept for backward compatibility with legacy results
            is_flat_structure = len(combined_participant_pool) > 0 and 'responses' not in combined_participant_pool[0]
            
            if is_flat_structure:
                # Convert flat structure to nested structure for evaluator
                print(f"    - Converting flat structure to nested structure for evaluator...")
                nested_participants = defaultdict(lambda: {"responses": [], "profile": {}})
                
                for item_response in combined_participant_pool:
                    p_id = item_response.get("participant_id", 0)
                    # Ensure profile is copied from the trial_info if available
                    if not nested_participants[p_id]["profile"] and "trial_info" in item_response and "profile" in item_response["trial_info"]:
                        nested_participants[p_id]["profile"] = item_response["trial_info"]["profile"]
                    # Add the response to the participant's responses list
                    nested_participants[p_id]["responses"].append(item_response)
                
                # Convert defaultdict to regular list with participant_id set
                combined_participant_pool = []
                for p_id, p_data in nested_participants.items():
                    p_data["participant_id"] = p_id
                    combined_participant_pool.append(p_data)
                
                print(f"    - Converted {len(nested_participants)} flat responses to {len(combined_participant_pool)} nested participants")
            
            n_runs = len(all_runs_data)
            print(f"    - Processing {n_runs} run(s) with {len(combined_participant_pool)} total participants...")
            
            print(f"  - Alignment evaluation...")
            full_raw_results = {"individual_data": combined_participant_pool}

            from evaluation.alignment import (
                DEFAULT_ALIGNMENT_SUBSET,
                compute_alignment_subset_aggregate,
                evaluate_alignment_study,
            )

            alignment_result = evaluate_alignment_study(study_id, full_raw_results)
            alignment_aggregate = compute_alignment_subset_aggregate(
                {study_id: alignment_result} if alignment_result else {},
                subset=DEFAULT_ALIGNMENT_SUBSET,
            )
            evaluation_payload: Dict[str, Any] = {
                "alignment_evaluation": alignment_result,
                "alignment_subset_aggregate": alignment_aggregate,
            }

            print(f"    Alignment evaluation complete")

            # ===== Calculate Usage Statistics =====
            total_prompt_tokens = 0
            total_completion_tokens = 0
            total_tokens = 0
            total_cost = 0.0
            
            # Check if data is flat or nested to iterate correctly
            is_flat_structure = len(combined_participant_pool) > 0 and 'responses' not in combined_participant_pool[0]
            
            if is_flat_structure:
                # Handle flat structure: each item is a trial response
                for resp in combined_participant_pool:
                    usage = resp.get('usage', {})
                    total_prompt_tokens += usage.get('prompt_tokens', 0) or 0
                    total_completion_tokens += usage.get('completion_tokens', 0) or 0
                    total_tokens += usage.get('total_tokens', 0) or 0
                    total_cost += usage.get('cost', 0.0) or 0.0
                n_participants_count = len(set(resp.get('participant_id') for resp in combined_participant_pool))
            else:
                # Handle nested structure: each item is a participant with multiple responses
                for participant in combined_participant_pool:
                    for resp in participant.get('responses', []):
                        usage = resp.get('usage', {})
                        total_prompt_tokens += usage.get('prompt_tokens', 0) or 0
                        total_completion_tokens += usage.get('completion_tokens', 0) or 0
                        total_tokens += usage.get('total_tokens', 0) or 0
                        total_cost += usage.get('cost', 0.0) or 0.0
                n_participants_count = len(combined_participant_pool)
            
            avg_tokens_per_participant = total_tokens / n_participants_count if n_participants_count > 0 else 0
            avg_cost_per_participant = total_cost / n_participants_count if n_participants_count > 0 else 0
            
            # ===== Calculate Final Failure Rate (from sanity check) =====
            final_failed = sanity_check_result.get('failed', 0)
            final_total = sanity_check_result.get('total_checked', 0)
            final_failure_rate = (final_failed / final_total * 100.0) if final_total > 0 else 0.0
            
            print(f"      Final failure rate: {final_failure_rate:.2f}% ({final_failed}/{final_total})")
            
            usage_stats_block = {
                'usage_stats': {
                    'total_prompt_tokens': total_prompt_tokens,
                    'total_completion_tokens': total_completion_tokens,
                    'total_tokens': total_tokens,
                    'total_cost': float(total_cost),
                    'avg_tokens_per_participant': float(avg_tokens_per_participant),
                    'avg_cost_per_participant': float(avg_cost_per_participant)
                },
                'failure_rates': {
                    'raw_failure_rate': float(raw_failure_rate),
                    'raw_failed': raw_failure_stats.get('raw_failed', 0),
                    'raw_total': raw_failure_stats.get('raw_total', 0),
                    'raw_failure_breakdown': raw_failure_stats.get('raw_failure_breakdown', {}),
                    'final_failure_rate': float(final_failure_rate),
                    'final_failed': final_failed,
                    'final_total': final_total
                }
            }

            evaluation_payload.update({
                'n_participants': len(combined_participant_pool),
                'n_runs': n_runs,
            })
            evaluation_payload.update(usage_stats_block)
            
            # Print usage summary
            print(f"      Usage: total {total_tokens} tokens, total cost ${total_cost:.4f}")
            print(f"      Average per participant: {avg_tokens_per_participant:.1f} tokens, ${avg_cost_per_participant:.6f}")
            
            json_path = cfg_dir / "evaluation_results.json"
            from evaluation.io_utils import atomic_write_json
            atomic_write_json(json_path, evaluation_payload, indent=2, ensure_ascii=False, encoding='utf-8', errors='replace')

            align_val = (alignment_result or {}).get("study_alignment_score")
            align_str = f"{align_val:.4f}" if align_val is not None else "N/A"
            print(f"    Results saved: alignment score = {align_str}")
            print(f"    Saved: {json_path.name}")
        
        return last_evaluator_path
