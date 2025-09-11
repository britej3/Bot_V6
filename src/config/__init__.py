# Import get_settings from the config.py module to resolve package/module naming conflict
import sys
import os
import importlib.util

# Load config.py directly to avoid circular import
config_module_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.py')
spec = importlib.util.spec_from_file_location("config_module", config_module_path)
config_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config_module)

# Export get_settings function
get_settings = config_module.get_settings

__all__ = ['get_settings']