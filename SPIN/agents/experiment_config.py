"""
Study configuration base class, registry, and PromptBuilder (merged from ``src/core/prompt_builder.py`` and ``study_config.py``).

PromptBuilder turns specification / materials into natural-language prompts for participants.
"""

import json
import re
from pathlib import Path
from typing import Dict, Any, List, Optional


class PromptBuilder:
    """
    Base class for building prompts from study specifications.
    
    Key responsibility: Transform technical specification.json
    into natural language prompts for LLM participants.
    """
    
    def __init__(self, study_path: Path):
        """
        Initialize prompt builder with study materials.
        
        Args:
            study_path: Path to study directory (e.g., data/studies/study_003/)
        """
        self.study_path = Path(study_path)
        self.materials_path = self.study_path / "materials"
        
        # Load specification
        with open(self.study_path / "specification.json", "r", encoding='utf-8', errors='replace') as f:
            self.specification = json.load(f)
        
        # Load instructions
        instructions_file = self.materials_path / "instructions.txt"
        if instructions_file.exists():
            with open(instructions_file, "r", encoding='utf-8', errors='replace') as f:
                self.instructions = f.read()
        else:
            self.instructions = None
        
        # Load custom system prompt template (optional)
        system_prompt_file = self.materials_path / "system_prompt.txt"
        if system_prompt_file.exists():
            with open(system_prompt_file, "r", encoding='utf-8', errors='replace') as f:
                self.system_prompt_template = f.read()
        else:
            self.system_prompt_template = None
    
    def build_system_prompt(self, participant_profile: Dict[str, Any] = None) -> Optional[str]:
        """
        Get the custom system prompt content if it exists.
        
        Note: Custom system prompt is appended as-is without template variable substitution.
        Age, gender, and other profile information are already included in the default
        system prompt, so the custom content should be additional instructions only.
        
        Args:
            participant_profile: Not used (kept for compatibility), custom prompt is used as-is
            
        Returns:
            Custom system prompt content string, or None if not provided
        """
        # Return custom system prompt template as-is (no template variable substitution)
        return self.system_prompt_template
    
    def get_system_prompt_template(self) -> Optional[str]:
        """
        Get the custom system prompt template if it exists.
        
        Returns:
            Custom system prompt template string, or None if not provided
        """
        return self.system_prompt_template
    
    def build_trial_prompt(self, trial_data: Dict[str, Any]) -> str:
        """
        Build the prompt for a single trial.
        
        Args:
            trial_data: Trial-specific information (stimuli, confederate responses, etc.)
                       May include 'participant_profile' for frame-specific prompts
            
        Returns:
            Complete trial prompt string
        """
        return self._build_generic_trial_prompt(trial_data)
    
    def get_instructions(self) -> str:
        """
        Get the experimental instructions.
        
        Returns:
            Instructions text (from materials/instructions.txt)
        """
        return self.instructions if self.instructions else "No instructions provided."
    
    def _fill_template(self, template: str, data: Dict[str, Any]) -> str:
        """
        Fill template with data using simple {{variable}} syntax.
        
        Supports:
        - {{variable}}: Simple substitution
        - {{object.key}}: Nested property access
        - {{#if variable}}...{{/if}}: Conditional blocks
        - {{#each array}}{{this}}{{/each}}: Loops (simplified)
        """
        result = template
        
        # Handle nested property access (e.g., {{comparison_lines.A}})
        nested_pattern = r'\{\{([\w.]+)\}\}'
        
        def replace_nested(match):
            path = match.group(1)
            parts = path.split('.')
            
            value = data
            for part in parts:
                if isinstance(value, dict) and part in value:
                    value = value[part]
                else:
                    return match.group(0)  # Keep original if not found
            
            return str(value)
        
        result = re.sub(nested_pattern, replace_nested, result)
        
        # Handle conditional blocks (simplified)
        # {{#if variable}}...{{/if}}
        if_pattern = r'\{\{#if\s+(\w+)\}\}(.*?)\{\{/if\}\}'
        
        def replace_if(match):
            var_name = match.group(1)
            content = match.group(2)
            # Check if variable exists and is truthy
            if var_name in data and data[var_name]:
                return content
            return ""
        
        result = re.sub(if_pattern, replace_if, result, flags=re.DOTALL)
        
        # Handle each loops (simplified)
        # {{#each array}}{{this}}{{/each}}
        each_pattern = r'\{\{#each\s+(\w+)\}\}(.*?)\{\{/each\}\}'
        
        def replace_each(match):
            var_name = match.group(1)
            content = match.group(2)
            
            if var_name not in data:
                return ""
            
            items = data[var_name]
            if isinstance(items, dict):
                # Dictionary: replace {{@key}} and {{this}}
                parts = []
                for key, value in items.items():
                    item_content = content.replace("{{@key}}", str(key))
                    item_content = item_content.replace("{{this}}", str(value))
                    parts.append(item_content)
                return "\n".join(parts)
            elif isinstance(items, list):
                # List: replace {{this}} and {{@index}}
                parts = []
                for idx, item in enumerate(items):
                    item_content = content.replace("{{@index}}", str(idx + 1))
                    item_content = item_content.replace("{{this}}", str(item))
                    parts.append(item_content)
                return "\n".join(parts)
            return ""
        
        result = re.sub(each_pattern, replace_each, result, flags=re.DOTALL)
        
        # Clean up any remaining unfilled placeholders
        result = re.sub(r'\{\{[^}]+\}\}', '', result)
        
        return result
    
    def _build_generic_system_prompt(self, profile: Dict[str, Any]) -> str:
        """Fallback generic system prompt."""
        age = profile.get('age', 'unknown age')
        gender = profile.get('gender', 'unspecified gender')
        
        return f"""You are participating in a psychology experiment as a real human participant.

Your identity: {age} years old, {gender}

Respond naturally as this person would. Do not explain your reasoning - just give direct responses as a real participant would."""
    
    def _build_generic_trial_prompt(self, trial_data: Dict[str, Any]) -> str:
        """Fallback generic trial prompt."""
        return f"Trial {trial_data.get('trial_number', '?')}: Please respond to the following stimulus."


def create_prompt_builder(study_path: Path) -> PromptBuilder:
    """
    Factory function to create PromptBuilder for a study.
    
    Args:
        study_path: Path to study directory
        
    Returns:
        PromptBuilder instance
    """
    return PromptBuilder(study_path)

# Convenience function for users
def get_prompt_builder(study_id: str, data_dir: Optional[str] = None) -> PromptBuilder:
    """
    Get prompt builder for a study by ID.
    
    Args:
        study_id: Study identifier (e.g., "study_003")
        data_dir: Path to data directory (parent of ``studies/``). Defaults to ``SPIN_DATA_DIR`` / sibling ``../data``.
        
    Returns:
        PromptBuilder instance
        
    Example:
        >>> builder = get_prompt_builder("study_003")
        >>> system_prompt = builder.build_system_prompt({"age": 20, "education": "university_student"})
        >>> trial_prompt = builder.build_trial_prompt({"trial_number": 1, ...})
    """
    from paths import spin_data_dir

    root = Path(data_dir) if data_dir else spin_data_dir()
    study_path = root / "studies" / study_id
    return create_prompt_builder(study_path)


from abc import ABC, abstractmethod


class BaseStudyConfig(ABC):
    """
    Base class for per-study configuration.

    Each study subclass implements:
    - create_trials(): build experiment trials
    - aggregate_results(): optional aggregation (default provided)
    - custom_scoring(): optional custom scoring
    """

    prompt_builder_class = PromptBuilder  # Default; subclasses may override
    
    def __init__(self, study_path: Path, specification: Dict[str, Any]):
        """
        Args:
            study_path: Study directory (e.g. data/studies/study_003/)
            specification: Parsed specification.json
        """
        self.study_path = Path(study_path)
        self.specification = specification
        self.study_id = specification["study_id"]
        
        # Initialize prompt builder with the configured class
        self.prompt_builder = self.prompt_builder_class(self.study_path)

    def load_material(self, sub_study_id: str) -> Dict[str, Any]:
        """Load JSON for sub_study_id from the materials directory."""
        file_path = self.study_path / "materials" / f"{sub_study_id}.json"
        if not file_path.exists():
            raise FileNotFoundError(f"Material not found: {file_path}")
        try:
            with open(file_path, "r", encoding='utf-8') as f:
                return json.load(f)
        except UnicodeDecodeError as e:
            # Try to detect encoding and provide helpful error
            import chardet
            with open(file_path, "rb") as f:
                raw = f.read()
                detected = chardet.detect(raw)
            raise UnicodeDecodeError(
                'utf-8', raw, e.start, e.end,
                f"File encoding issue. Detected: {detected.get('encoding', 'unknown')} "
                f"(confidence: {detected.get('confidence', 0):.2f}). "
                f"Please ensure the file is UTF-8 encoded."
            )

    def load_metadata(self) -> Dict[str, Any]:
        """Load metadata.json."""
        file_path = self.study_path / "metadata.json"
        with open(file_path, "r", encoding='utf-8') as f:
            return json.load(f)

    def load_specification(self) -> Dict[str, Any]:
        """Load specification.json."""
        file_path = self.study_path / "specification.json"
        with open(file_path, "r", encoding='utf-8') as f:
            return json.load(f)

    def load_ground_truth(self) -> Dict[str, Any]:
        """Load ground_truth.json."""
        file_path = self.study_path / "ground_truth.json"
        with open(file_path, "r", encoding='utf-8') as f:
            return json.load(f)

    def extract_numeric(self, text: str, default: float = 0.0) -> float:
        """Extract the first number from text (supports negatives and decimals)."""
        if text is None: return default
        import re
        match = re.search(r"(-?\d+\.?\d*)", str(text))
        return float(match.group(1)) if match else default

    def extract_choice(self, text: str, options: List[str] = None) -> Optional[int]:
        """Extract a choice index (0, 1, 2, ...) from text."""
        if text is None: return None
        import re
        text_s = str(text).strip()
        
        # 1. Match option text if options are provided
        if options:
            for i, opt in enumerate(options):
                if opt.lower() in text_s.lower():
                    return i
        
        # 2. Single-letter choices like "A", "Choice A", "(A)"
        match = re.search(r"\b([A-Z])\b", text_s.upper())
        if match:
            # A->0, B->1...
            return ord(match.group(1)) - ord('A')
            
        return None
    
    @abstractmethod
    def create_trials(self, n_trials: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Build trials from the specification.

        Args:
            n_trials: Number of trials (None = specification default)

        Returns:
            List of trial dicts, each with at least:
            - trial_number: int
            - study_type: str (e.g. "framing_effect")
            - trial_type: str (e.g. "practice", "critical", "neutral")
            - additional study-specific fields
        """
        raise NotImplementedError
    
    def get_prompt_builder(self) -> PromptBuilder:
        """Return the prompt builder."""
        return self.prompt_builder
    
    def get_instructions(self) -> str:
        """Return experimental instructions text."""
        return self.prompt_builder.get_instructions()
    
    def aggregate_results(self, raw_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Aggregate experiment results.

        Default: return ParticipantPool.run_experiment() output unchanged.
        Override to add custom summaries or statistics.

        Args:
            raw_results: Output from ParticipantPool.run_experiment()

        Returns:
            Aggregated dict, e.g.:
            {
                "descriptive_statistics": {...},
                "inferential_statistics": {...},
                "individual_data": [...],
                "raw_responses": [...]
            }
        """
        return raw_results
    
    def custom_scoring(
        self,
        results: Dict[str, Any],
        ground_truth: Dict[str, Any]
    ) -> Optional[Dict[str, float]]:
        """
        Optional custom scoring.

        Override when a study needs non-default scoring.

        Args:
            results: Output of aggregate_results()
            ground_truth: ground_truth.json content

        Returns:
            None (use default Scorer) or a score dict:
            {
                "test_name_1": 0.8,
                "test_name_2": 0.6,
                ...
            }
        """
        return None
    
    def get_n_participants(self) -> int:
        """Participant count from specification."""
        return self.specification["participants"]["n"]
    
    def get_study_type(self) -> str:
        """Study type string."""
        return self.specification.get("study_type", self.study_id)
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(study_id='{self.study_id}')"


class StudyConfigRegistry:
    """
    Registry of study configuration classes.

    Classes self-register via the decorator below.
    """
    
    _configs: Dict[str, type] = {}
    
    @classmethod
    def register(cls, study_id: str):
        """
        Decorator to register a study config class.

        Usage:
            @StudyConfigRegistry.register("study_003")
            class Study003Config(BaseStudyConfig):
                ...
        """
        def decorator(config_class):
            cls._configs[study_id] = config_class
            return config_class
        return decorator
    
    @classmethod
    def get_config_class(cls, study_id: str) -> Optional[type]:
        """Return the registered class for study_id, if any."""
        return cls._configs.get(study_id)
    
    @classmethod
    def create_config(
        cls, 
        study_id: str, 
        study_path: Path, 
        specification: Dict[str, Any]
    ) -> Optional[BaseStudyConfig]:
        """
        Instantiate the config for study_id.

        Args:
            study_id: Study identifier
            study_path: Study directory
            specification: specification.json content

        Returns:
            Config instance, or None if no class is registered
        """
        config_class = cls.get_config_class(study_id)
        if config_class:
            return config_class(study_path, specification)
        return None
    
    @classmethod
    def list_registered_studies(cls) -> List[str]:
        """List registered study IDs."""
        return list(cls._configs.keys())


def get_study_config(
    study_id: str, 
    study_path: Path, 
    specification: Dict[str, Any]
) -> BaseStudyConfig:
    """
    Factory: build the config instance for study_id.

    Args:
        study_id: Study ID (e.g. "study_003")
        study_path: Study directory
        specification: specification.json content

    Returns:
        Study config instance
    """
    # Dynamic import triggers @StudyConfigRegistry.register (former ``src/studies`` lives in material_studies)
    import evaluation.material_studies  # noqa: F401

    config = StudyConfigRegistry.create_config(study_id, study_path, specification)
    
    if config is None:
        raise ValueError(
            f"No configuration found for {study_id}. "
            f"Available: {StudyConfigRegistry.list_registered_studies()}"
        )
    
    return config
