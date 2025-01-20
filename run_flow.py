# accept a config
# select a flow accordingly
# read config, call the flow, send config through the flow call
# import yaml
# import json
# import argparse
# from importlib import import_module

# parser = argparse.ArgumentParser()
# parser.add_argument("--config", type=str,
#                     help="path to a config file")
# args = parser.parse_args()

# with open(args.config, 'r') as f:
#     if args.config.endswith(('.yml', '.yaml')):
#         config = yaml.safe_load(f)
#     else:
#         config = json.load(f)

# flow_module = import_module(config['flow_module'])
# flow_module.start(config)


import yaml
import json
import argparse
from importlib import import_module
import sys

def load_config(file_path):
    """
    Load configuration from a YAML or JSON file.
    """
    try:
        with open(file_path, 'r') as f:
            if file_path.endswith(('.yml', '.yaml')):
                return yaml.safe_load(f)
            elif file_path.endswith('.json'):
                return json.load(f)
            else:
                raise ValueError("Unsupported file format. Use YAML or JSON.")
    except Exception as e:
        print(f"Error loading config: {e}")
        sys.exit(1)

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run a flow with a given config.")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to a config file (YAML or JSON).")
    parser.add_argument("--model_type", type=str, default=None,
                        help="Override model type in the config (optional).")
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Override model_type if provided
    if args.model_type:
        if 'model' in config and isinstance(config['model'], dict):
            config['model']['model_type'] = args.model_type
            print(f"Overriding model_type to: {args.model_type}")
        else:
            print("Warning: Cannot override model_type as 'model' key is missing in the config.")

    # Ensure 'flow_module' exists in the config
    if 'flow_module' not in config:
        print("Error: 'flow_module' not found in config.")
        sys.exit(1)

    # Import and execute the specified flow
    try:
        flow_module = import_module(config['flow_module'])
        flow_module.start(config)
    except ModuleNotFoundError:
        print(f"Error: Module '{config['flow_module']}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error executing flow: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
