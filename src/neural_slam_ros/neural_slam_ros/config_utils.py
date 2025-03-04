import yaml
import logging

logger = logging.getLogger(__name__)

# Custom OpenCV matrix class for YAML parsing
class OpenCVMatrix:
    def __init__(self, rows=4, cols=4, dt="f", data=None):
        self.rows = rows
        self.cols = cols
        self.dt = dt
        self.data = data if data is not None else []

def opencv_matrix_constructor(loader, node):
    """Parse a YAML map with tag !!opencv-matrix into an OpenCVMatrix object."""
    mapping = loader.construct_mapping(node, deep=True)
    return OpenCVMatrix(**mapping)

def load_drone_config(path):
    """
    Load drone configuration from YAML file, handling OpenCV matrix format.
    
    Args:
        path: Path to the drone config YAML file
        
    Returns:
        Dictionary containing drone configuration
    """
    # Create a custom loader that can handle !!opencv-matrix
    class OpenCVLoader(yaml.SafeLoader):
        pass
    
    # Register the constructor for !!opencv-matrix
    OpenCVLoader.add_constructor('!!opencv-matrix', opencv_matrix_constructor)
    
    try:
        with open(path, 'r') as f:
            content = f.read()
            # Skip the %YAML:1.0 directive if present
            if content.strip().startswith('%YAML'):
                # Find the first newline and skip everything before it
                first_newline = content.find('\n')
                if first_newline != -1:
                    content = content[first_newline+1:]
            return yaml.load(content, Loader=OpenCVLoader)
    except FileNotFoundError:
        logger.error(f"Could not find config file at {path}")
        return None
    except Exception as e:
        logger.error(f"Failed to load config file: {e}")
        return None 