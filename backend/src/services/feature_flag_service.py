import json
from pathlib import Path
from typing import Dict, List, Optional

from backend.src.models.feature_flag import FeatureFlag

CONFIG_PATH = Path(__file__).parent.parent / "config" / "feature_flags.json"

class FeatureFlagService:
    def __init__(self):
        self._flags: Dict[str, FeatureFlag] = {}
        self.load_flags()

    def load_flags(self):
        if not CONFIG_PATH.exists():
            self._flags = {}
            return

        with open(CONFIG_PATH, "r") as f:
            data = json.load(f)
            for name, flag_data in data.items():
                self._flags[name] = FeatureFlag(name=name, **flag_data)

    def get_all_flags(self) -> List[FeatureFlag]:
        return list(self._flags.values())

    def get_flag(self, name: str) -> Optional[FeatureFlag]:
        return self._flags.get(name)

    def update_flag(self, name: str, enabled: bool) -> Optional[FeatureFlag]:
        if name not in self._flags:
            return None

        self._flags[name].enabled = enabled
        self._save_flags()
        return self._flags[name]

    def _save_flags(self):
        data = {
            name: flag.model_dump(exclude={'name'})
            for name, flag in self._flags.items()
        }
        with open(CONFIG_PATH, "w") as f:
            json.dump(data, f, indent=2)

feature_flag_service = FeatureFlagService()
