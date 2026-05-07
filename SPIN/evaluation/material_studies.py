"""
P_M / S_T: study configuration (trials + prompts) and response parsers for sanity checks.
"""
import json
import re
import random
from pathlib import Path
from typing import Dict, Any, List, Optional

from agents.experiment_config import BaseStudyConfig, StudyConfigRegistry, PromptBuilder
from paths import study_root
# --- Study configs (P_M, S_T) ---


class PM_CustomPromptBuilder(PromptBuilder):
    def __init__(self, study_path: Path):
        super().__init__(study_path)

    def build_trial_prompt(self, trial_metadata: Dict[str, Any]) -> str:
        items = trial_metadata.get("items", [])
        sub_id = trial_metadata.get("sub_study_id")
        
        prompt = "Please answer the following questions.\n\n"
        
        if sub_id == "study_4_keg_ban_alienation":
            prompt += "The university has recently instituted a new policy banning kegs of beer on campus. Please respond to the following questions regarding this policy.\n\n"

        q_indices = []
        for i, item in enumerate(items):
            q_idx = f"Q{i+1}"
            q_indices.append(q_idx)
            prompt += f"{q_idx}: {item['question']}\n"
            if "options" in item and item["options"] is not None:
                # Check if options is iterable (list/tuple) before joining
                if isinstance(item["options"], (list, tuple)) and len(item["options"]) > 0:
                    prompt += f"Options: {', '.join(str(opt) for opt in item['options'])}\n"
            prompt += "\n"

        spec_format = ", ".join([f"{idx}=<number>" for idx in q_indices])
        prompt += f"RESPONSE_SPEC: Provide your answers in the following format: {spec_format}"
        
        return prompt

@StudyConfigRegistry.register("P_M")
class StudyPMConfig(BaseStudyConfig):
    prompt_builder_class = PM_CustomPromptBuilder
    PROMPT_VARIANT = "v1"

    def __init__(self, study_path: Path, specification: Dict[str, Any]):
        super().__init__(study_path, specification)

    def create_trials(self, n_trials: int = None) -> List[Dict[str, Any]]:
        trials = []
        spec = self.specification
        
        # Study 1: Pluralistic Ignorance regarding Alcohol Habits
        sub_id_1 = "study_1_comfort_estimation"
        material_1 = self.load_material(sub_id_1)
        n_1 = spec["participants"]["by_sub_study"].get(sub_id_1, {}).get("n", 50)
        if n_1 == 0: n_1 = 50
        
        for _ in range(n_1):
            trials.append({
                "sub_study_id": sub_id_1,
                "items": material_1["items"],
                "profile": {"age": random.randint(18, 22), "gender": random.choice(["Male", "Female"])},
                "variant": self.PROMPT_VARIANT
            })

        # Study 2: Pluralistic Ignorance regarding Friends and Order Effects
        sub_id_2 = "study_2_order_and_friend_comparison"
        material_2 = self.load_material(sub_id_2)
        n_2 = spec["participants"]["by_sub_study"].get(sub_id_2, {}).get("n", 50)
        if n_2 == 0: n_2 = 50
        
        # In Study 2, questions were Self, Average Student, and Friends.
        # Order of Self and Average Student was manipulated.
        for i in range(n_2):
            items = material_2["items"].copy()
            # items[0] is Self, items[1] is Average Student, items[2] is Friend
            order = "self_first" if i < n_2 // 2 else "average_first"
            
            if order == "average_first":
                # Swap first two items
                items[0], items[1] = items[1], items[0]
            
            trials.append({
                "sub_study_id": sub_id_2,
                "items": items,
                "order_condition": order,
                "profile": {"age": random.randint(18, 22), "gender": random.choice(["Male", "Female"])},
                "variant": self.PROMPT_VARIANT
            })

        # Study 4: Pluralistic Ignorance and Campus Alienation regarding Keg Ban
        sub_id_4 = "study_4_keg_ban_alienation"
        material_4 = self.load_material(sub_id_4)
        n_4 = spec["participants"]["by_sub_study"].get(sub_id_4, {}).get("n", 50)
        if n_4 == 0: n_4 = 50
        
        for _ in range(n_4):
            trials.append({
                "sub_study_id": sub_id_4,
                "items": material_4["items"],
                "profile": {"age": random.randint(18, 22), "gender": random.choice(["Male", "Female"])},
                "variant": self.PROMPT_VARIANT
            })

        return trials

    def extract_results(self, response_text: str, trial_metadata: Dict[str, Any]) -> Dict[str, Any]:
        results = {}
        items = trial_metadata.get("items", [])
        
        for i, item in enumerate(items):
            q_idx = f"Q{i+1}"
            val = self.extract_numeric(response_text, q_idx)
            
            # Map back to original question type for analysis
            # Since items were copied/reordered in Study 2, we use the 'id' from the item
            item_id = item.get("id")
            results[item_id] = val
            
        return results

    def dump_prompts(self, output_dir: str):
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        # Create one sample trial per sub-study for dumping
        sub_studies = ["study_1_comfort_estimation", "study_2_order_and_friend_comparison", "study_4_keg_ban_alienation"]
        
        for sub_id in sub_studies:
            all_trials = self.create_trials(n_trials=2)
            trial = next(t for t in all_trials if t["sub_study_id"] == sub_id)
            prompt = self.prompt_builder.build_trial_prompt(trial)
            with open(output_path / f"P_M_{sub_id}.txt", "w", encoding="utf-8") as f:
                f.write(prompt)

class ST_CustomPromptBuilder(PromptBuilder):
    def __init__(self, study_path: Path):
        super().__init__(study_path)

    def build_trial_prompt(self, trial_metadata: Dict[str, Any]) -> str:
        """
        Builds the prompt for a single participant based on the sub-study.
        Each participant receives all items for their assigned sub-study in a single trial.
        """
        instructions = trial_metadata.get("instructions", "")
        items = trial_metadata.get("items", [])
        variant = trial_metadata.get("variant", "v1")
        
        prompt = f"{instructions}\n\n"

        q_indices = []
        for idx, item in enumerate(items):
            q_idx = f"Q{idx + 1}"
            q_indices.append(q_idx)
            
            # Format multiple choice options as A/B/C
            if item.get("type") == "multiple_choice" and "options" in item:
                options = item.get("options", [])
                option_letters = [chr(65 + i) for i in range(len(options))]  # A, B, C, ...
                options_str = "\n".join([f"{letter}) {opt}" for letter, opt in zip(option_letters, options)])
                prompt += f"{q_idx}: {item['question']}\n{options_str}\n\n"
                # Store option mapping for response parsing
                item["option_map"] = {letter: opt for letter, opt in zip(option_letters, options)}
                item["option_letters"] = option_letters
            else:
                # For non-multiple choice items, use original format
                options_str = "\n".join([f"- {opt}" for opt in item.get("options", [])])
                prompt += f"{q_idx}: {item['question']}\nOptions:\n{options_str}\n\n"
        
        # Define the expected response format
        response_specs = []
        for idx, item in enumerate(items):
            q_idx = f"Q{idx + 1}"
            if item.get("type") == "multiple_choice" and "option_letters" in item:
                # Format as Q1=<A/B/C>
                response_specs.append(f"{q_idx}=<{'/'.join(item['option_letters'])}>")
            else:
                response_specs.append(f"{q_idx}=<choice>")
        
        prompt += f"RESPONSE_SPEC: Please provide your answers in the following format: {', '.join(response_specs)}\n"
        
        return prompt

@StudyConfigRegistry.register("S_T")
class StudySTConfig(BaseStudyConfig):
    prompt_builder_class = ST_CustomPromptBuilder
    PROMPT_VARIANT = "v1"

    def __init__(self, study_path: Path, specification: Dict[str, Any]):
        super().__init__(study_path, specification)

    def create_trials(self, n_trials: int = None) -> List[Dict[str, Any]]:
        trials = []
        sub_studies = ["pd_triad_tasks", "newcombs_computer_task", "pd_info_seeking_variation"]
        
        # Map sub_study_id to specification experiment names
        sub_to_experiment = {
            "pd_triad_tasks": "Experiment 1",
            "newcombs_computer_task": "Experiment 2",
            "pd_info_seeking_variation": "Experiment 3"
        }
        
        # Default sample sizes based on human experiments
        default_ns = {
            "pd_triad_tasks": 80,  # Experiment 1: 80 participants
            "newcombs_computer_task": 40,  # Experiment 2: 40 participants
            "pd_info_seeking_variation": 80  # Experiment 3: assume similar to Exp 1
        }
        
        # Load participant counts from specification
        spec = self.load_specification()
        n_by_sub = spec.get("participants", {}).get("by_sub_study", {})
        
        for sub_id in sub_studies:
            material = self.load_material(sub_id)
            
            # Determine n for this sub-study
            if n_trials is not None:
                n = n_trials
            else:
                # Try to get from specification using experiment name
                exp_name = sub_to_experiment.get(sub_id)
                if exp_name and exp_name in n_by_sub:
                    n = n_by_sub[exp_name].get("n", 0)
                else:
                    n = 0
                
                # If not found or 0, use default based on human experiment
                if n == 0:
                    n = default_ns.get(sub_id, 50)

            for _ in range(n):
                # In this study, a 'trial' is a single participant completing a whole task set
                trials.append({
                    "sub_study_id": sub_id,
                    "instructions": material.get("instructions", ""),
                    "items": material.get("items", []),
                    "variant": self.PROMPT_VARIANT
                })
        
        return trials

    def parse_responses(self, trial_metadata: Dict[str, Any], response_text: str) -> Dict[str, Any]:
        items = trial_metadata.get("items", [])
        results = {}
        
        # First parse Q1=value, Q2=value format
        parsed_responses = {}
        pattern = re.compile(r"(Q\d+(?:\.\d+)?)\s*=\s*([^,\n\s]+)")
        for k, v in pattern.findall(response_text):
            parsed_responses[k.strip()] = v.strip()
        
        for idx, item in enumerate(items):
            q_idx = f"Q{idx + 1}"
            item_id = item.get("id")
            
            # Get the value for this Q index
            choice_text = parsed_responses.get(q_idx, "")
            
            if choice_text:
                # For multiple choice, extract_choice can handle both A/B/C and option text
                choice_idx = self.extract_choice(choice_text, item.get("options", []))
                if choice_idx is not None:
                    # Return the actual option text, not the index
                    options = item.get("options", [])
                    if choice_idx < len(options):
                        results[item_id] = options[choice_idx]
                    else:
                        results[item_id] = choice_text  # Fallback to raw text
                else:
                    results[item_id] = choice_text  # Fallback to raw text if extraction fails
        
        return results

    def dump_prompts(self, output_dir: str):
        # Create one sample prompt for each sub-study
        sub_studies = ["pd_triad_tasks", "newcombs_computer_task", "pd_info_seeking_variation"]
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for sub_id in sub_studies:
            material = self.load_material(sub_id)
            trial = {
                "sub_study_id": sub_id,
                "instructions": material.get("instructions", ""),
                "items": material.get("items", []),
                "variant": self.PROMPT_VARIANT
            }
            prompt = self.prompt_builder.build_trial_prompt(trial)
            with open(output_path / f"S_T_{sub_id}_prompt.txt", "w") as f:
                f.write(prompt)

# --- Evaluators ---

def _pm_parse_agent_responses(response_text: str) -> Dict[str, str]:
    """Parse Qk=<value> or Qk: <value> or Qk.n=<value> format from response text."""
    results = {}
    # Supports both = and : separators
    pattern = re.compile(r"(Q\d+(?:\.\d+)?)\s*[:=]\s*([^,\n\s]+)")
    for k, v in pattern.findall(response_text):
        results[k.strip()] = v.strip()
    return results

def _pm_get_required_q_numbers(trial_info: Dict[str, Any]) -> set:
    """
    Extract all required Q labels from trial_info.
    Study_008: read q_idx_choice from items.
    """
    required = set()
    items = trial_info.get("items", [])
    
    for item in items:
        q_idx = item.get("q_idx_choice")
        if q_idx:
            # If q_idx already has a "Q" prefix, keep it; otherwise add "Q"
            if isinstance(q_idx, str) and q_idx.startswith("Q"):
                required.add(q_idx)
            else:
                required.add(f"Q{q_idx}")
        # Infer from item index when q_idx_choice is absent
        elif not required:
            for idx, _ in enumerate(items):
                required.add(f"Q{idx + 1}")
            break
    
    return required

def _st_parse_agent_responses(response_text: str) -> Dict[str, str]:
    """Parse Qk=<value> or Qk: <value> or Qk.n=<value> format from raw response text."""
    results = {}
    # Matches Q1=choice, Q1: choice, Q2=choice, etc.
    # Supports both = and : separators
    pattern = re.compile(r"(Q\d+(?:\.\d+)?)\s*[:=]\s*([^,\n\s]+)")
    for k, v in pattern.findall(response_text):
        results[k.strip()] = v.strip().lower()
    return results

def _st_get_required_q_numbers(trial_info: Dict[str, Any]) -> set:
    """
    Extract all required Q labels from trial_info.
    Study_010: one Q per item using index+1 (Q1..Qn for items[0]..items[n-1]).
    """
    required = set()
    items = trial_info.get("items", [])
    
    # Study_010: Qk maps to items[k-1]
    for idx, item in enumerate(items):
        required.add(f"Q{idx + 1}")
    
    return required


# --- Public API for pipeline / evaluator_runner / sanity_check ---

EVALUATOR_MODULE_PATH = Path(__file__).resolve()
BUNDLED_STUDY_IDS = frozenset({"P_M", "S_T"})


def is_bundled_evaluator(study_id: str) -> bool:
    return study_id in BUNDLED_STUDY_IDS


def get_sanity_evaluator_module(study_id: str):
    """Namespace with ``parse_agent_responses`` and ``get_required_q_numbers`` for sanity_check."""
    import types

    if study_id == "P_M":
        m = types.SimpleNamespace()
        m.parse_agent_responses = _pm_parse_agent_responses
        m.get_required_q_numbers = _pm_get_required_q_numbers
        return m
    if study_id == "S_T":
        m = types.SimpleNamespace()
        m.parse_agent_responses = _st_parse_agent_responses
        m.get_required_q_numbers = _st_get_required_q_numbers
        return m
    return None

__all__ = [
    "StudyPMConfig",
    "StudySTConfig",
    "get_sanity_evaluator_module",
    "EVALUATOR_MODULE_PATH",
    "is_bundled_evaluator",
    "BUNDLED_STUDY_IDS",
]
