import os
import yaml

class Config:
    def __init__(self, config_path=None):
        if config_path is None:
            # Assume we are in src/ and config is in ../config/config.yaml
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            config_path = os.path.join(base_dir, "config", "config.yaml")
        
        self.config_path = config_path
        self.data = {}
        self.load()

    def load(self):
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found at: {self.config_path}")
        
        with open(self.config_path, "r", encoding="utf-8") as f:
            self.data = yaml.safe_load(f)

    def get(self, key, default=None):
        keys = key.split(".")
        val = self.data
        for k in keys:
            if isinstance(val, dict) and k in val:
                val = val[k]
            else:
                return default
        return val

    @property
    def project_name(self):
        return self.get("system.project_name", "UIDAI Risk Engine")

    @property
    def version(self):
        return self.get("system.version", "1.0.0")

    def get_absolute_path(self, path_key):
        rel_path = self.get(f"paths.{path_key}")
        if rel_path is None:
            return None
        
        # Resolve path relative to project root
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return os.path.abspath(os.path.join(base_dir, rel_path))
