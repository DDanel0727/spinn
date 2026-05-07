"""
HumanStudyBench data loading: exceptions, Study model, registry access.

Merged from ``src/core/exceptions.py``, ``study.py``, and ``benchmark.py``.
"""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


class HumanStudyBenchError(Exception):
    """Base exception for HumanStudyBench."""
    pass


class StudyNotFoundError(HumanStudyBenchError):
    """Raised when a requested study cannot be found."""
    pass


class ValidationError(HumanStudyBenchError):
    """Raised when data validation fails."""
    pass


class SchemaError(HumanStudyBenchError):
    """Raised when schema validation fails."""
    pass


class AgentError(HumanStudyBenchError):
    """Raised when agent execution fails."""
    pass


class ConfigurationError(HumanStudyBenchError):
    """Raised when configuration is invalid."""
    pass


class DataLoadError(HumanStudyBenchError):
    """Raised when data cannot be loaded."""
    pass


@dataclass
class Study:
    """Represents a single human study in the benchmark."""
    
    id: str
    metadata: Dict[str, Any]
    specification: Dict[str, Any]
    ground_truth: Dict[str, Any]
    materials_path: Path
    
    # Pass thresholds
    PASS_THRESHOLD = 0.70  # 70% - minimum passing score for a study
    HIGH_QUALITY_THRESHOLD = 0.85  # 85% - high quality replication
    PERFECT_THRESHOLD = 1.00  # 100% - perfect replication
    
    @classmethod
    def load(cls, study_path: Path) -> "Study":
        """
        Load a study from disk.
        
        Args:
            study_path: Path to study directory
            
        Returns:
            Study object
            
        Raises:
            DataLoadError: If study files cannot be loaded
        """
        study_path = Path(study_path)
        
        if not study_path.exists():
            raise DataLoadError(f"Study directory not found: {study_path}")
        
        study_id = study_path.name
        
        try:
            # Load metadata
            with open(study_path / "metadata.json", "r", encoding='utf-8', errors='replace') as f:
                metadata = json.load(f)
            
            # Load specification
            with open(study_path / "specification.json", "r", encoding='utf-8', errors='replace') as f:
                specification = json.load(f)
            
            # Load ground truth
            with open(study_path / "ground_truth.json", "r", encoding='utf-8', errors='replace') as f:
                ground_truth = json.load(f)
            
            materials_path = study_path / "materials"
            
            return cls(
                id=study_id,
                metadata=metadata,
                specification=specification,
                ground_truth=ground_truth,
                materials_path=materials_path
            )
        
        except FileNotFoundError as e:
            raise DataLoadError(f"Required file missing in study {study_id}: {e}")
        except json.JSONDecodeError as e:
            raise DataLoadError(f"Invalid JSON in study {study_id}: {e}")
        except Exception as e:
            raise DataLoadError(f"Error loading study {study_id}: {e}")
    
    def get_validation_criteria(self) -> List[Dict[str, Any]]:
        """
        Get validation criteria for this study.
        
        Returns:
            List of validation test dictionaries
        """
        return self.ground_truth["validation_criteria"]["required_tests"]
    
    def get_materials(self, material_type: Optional[str] = None) -> Path:
        """
        Get path to study materials.
        
        Args:
            material_type: Type of material (e.g., 'stimuli', 'instructions')
                          If None, returns base materials path
        
        Returns:
            Path to requested materials
        """
        if material_type is None:
            return self.materials_path
        return self.materials_path / material_type
    
    def validate(self) -> bool:
        """
        Validate study data integrity.
        
        Returns:
            True if valid
            
        Raises:
            ValidationError: If validation fails
        """
        # Check IDs match
        if self.metadata["id"] != self.id:
            raise ValidationError(f"Metadata ID mismatch: {self.metadata['id']} != {self.id}")
        
        if self.specification["study_id"] != self.id:
            raise ValidationError(f"Specification ID mismatch: {self.specification['study_id']} != {self.id}")
        
        if self.ground_truth["study_id"] != self.id:
            raise ValidationError(f"Ground truth ID mismatch: {self.ground_truth['study_id']} != {self.id}")
        
        # Check materials directory exists
        if not self.materials_path.exists():
            raise ValidationError(f"Materials directory not found: {self.materials_path}")
        
        return True
    
    def get_domain(self) -> str:
        """Get study domain."""
        return self.metadata["domain"]
    
    def get_difficulty(self) -> str:
        """Get study difficulty level."""
        return self.metadata["difficulty"]
    
    def get_tags(self) -> List[str]:
        """Get study tags."""
        return self.metadata.get("tags", [])
    
    def get_pass_threshold(self) -> float:
        """Get the pass threshold for this study (can be customized per study)."""
        return self.ground_truth.get("validation_criteria", {}).get(
            "pass_threshold", 
            self.PASS_THRESHOLD
        )
    
    def evaluate_pass_status(self, score: float) -> Dict[str, Any]:
        """
        Evaluate if a score passes this study.
        
        Args:
            score: Overall score for this study (0.0 to 1.0)
        
        Returns:
            Dictionary with pass evaluation:
            {
                "passed": bool,
                "grade": str,  # "fail", "pass", "high_quality", "perfect"
                "score": float,
                "threshold": float,
                "margin": float  # score - threshold
            }
        """
        threshold = self.get_pass_threshold()
        
        if score >= self.PERFECT_THRESHOLD:
            grade = "perfect"
            passed = True
        elif score >= self.HIGH_QUALITY_THRESHOLD:
            grade = "high_quality"
            passed = True
        elif score >= threshold:
            grade = "pass"
            passed = True
        else:
            grade = "fail"
            passed = False
        
        return {
            "passed": passed,
            "grade": grade,
            "score": score,
            "threshold": threshold,
            "margin": score - threshold
        }
    
    def __repr__(self) -> str:
        return f"Study(id='{self.id}', title='{self.metadata.get('title', 'Unknown')}')"
    
    def __str__(self) -> str:
        return f"{self.id}: {self.metadata.get('title', 'Unknown')}"


class HumanStudyBench:
    """Load studies from a HumanStudyBench-style data directory (registry + per-study folders)."""

    def __init__(self, data_dir: str | Path, config: Optional[Dict[str, Any]] = None):
        self.data_dir = Path(data_dir)
        self.studies_dir = self.data_dir / "studies"
        self.schemas_dir = self.data_dir / "schemas"
        self.config = config or {}
        self.registry = self._load_registry()
        self.studies: Dict[str, Study] = {}

    def _load_registry(self) -> Dict[str, Any]:
        registry_path = self.data_dir / "registry.json"
        if not registry_path.exists():
            raise DataLoadError(f"Registry not found: {registry_path}")
        with open(registry_path, "r", encoding="utf-8", errors="replace") as f:
            return json.load(f)

    def load_study(self, study_id: str) -> Study:
        if study_id in self.studies:
            return self.studies[study_id]
        if self._get_study_info(study_id) is None:
            raise StudyNotFoundError(f"Study '{study_id}' not found in registry")
        study_path = self.studies_dir / study_id
        if not study_path.exists():
            raise StudyNotFoundError(f"Study directory not found: {study_path}")
        study = Study.load(study_path)
        self.studies[study_id] = study
        return study

    def _get_study_info(self, study_id: str) -> Optional[Dict[str, Any]]:
        for study_info in self.registry["studies"]:
            if study_info["id"] == study_id:
                return study_info
        return None

    def get_registry(self) -> Dict[str, Any]:
        return self.registry

    def __repr__(self) -> str:
        return (
            f"HumanStudyBench(total_studies={self.registry['total_studies']}, "
            f"version={self.registry['version']})"
        )
