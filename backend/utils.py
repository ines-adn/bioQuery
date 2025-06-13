import json
import logging


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_config(config_file="config.json"):
    """Loads config from a JSON file."""

    try:
        with open(config_file, "r") as file:
            config = json.load(file)
            return config
    
    except FileNotFoundError:
        logger.error(f"The file {config_file} was not found.")
        return {}
    except json.JSONDecodeError:
        logger.error(f"Error reading the file {config_file}.")
        return {}