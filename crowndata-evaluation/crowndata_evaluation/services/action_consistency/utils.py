import json
import numpy as np


# TODO: move it to common utils
def read_json_cartesian_pose(file_path: str) -> np.ndarray:
    """
    Reads a JSON file containing Cartesian pose data and returns the data as a NumPy array.

    The JSON file is expected to have entries for `x`, `y`, `z`, `roll`, `pitch`, and `yaw` values.
    Each entry in the list should be a dictionary with these keys.

    Parameters
    ----------
    file_path : str
        Path to the JSON file to be read.

    Returns
    -------
    np.ndarray
        A NumPy array containing the x, y, z, roll, pitch, and yaw values for each entry in the JSON file.

    Raises
    ------
    TypeError
        If `file_path` is not a string or if the output is not a NumPy array.
    ValueError
        If the JSON data is invalid, not a list of dictionaries, or if any of the data cannot be converted to floats.

    Examples
    --------
    >>> data = read_json_cartesian_pose("pose_data.json")
    >>> print(data.shape)
    (100, 6)  # Assuming 100 entries with x, y, z, roll, pitch, and yaw values.
    """

    # Input type check
    if not isinstance(file_path, str):
        raise TypeError(
            f"Expected `file_path` to be a string, but got {type(file_path).__name__}."
        )

    # Load the JSON file
    try:
        with open(file_path, "r") as file:
            data = json.load(file)
    except json.JSONDecodeError as e:
        raise ValueError(f"Error decoding JSON from file: {e}")

    # Validate that data is a list of dictionaries
    if not isinstance(data, list):
        raise ValueError("Expected JSON data to be a list of dictionaries.")

    # Create NumPy array from the extracted values
    try:
        xyzrpy_array = np.array(
            [
                [
                    float(entry.get("x")),
                    float(entry.get("y")),
                    float(entry.get("z")),
                    float(entry.get("roll")),
                    float(entry.get("pitch")),
                    float(entry.get("yaw")),
                ]
                for entry in data
                if isinstance(entry, dict)  # Ensure each entry is a dictionary
            ]
        )
    except (TypeError, ValueError) as e:
        raise ValueError(f"Error processing JSON data to floats: {e}")

    # Output type check
    if not isinstance(xyzrpy_array, np.ndarray):
        raise TypeError(
            f"Expected output to be a NumPy array, but got {type(xyzrpy_array).__name__}."
        )

    return xyzrpy_array


data = read_json_cartesian_pose(
    "crowndata-evaluation/example_data/data/droid_00000000/trajectories/cartesian_position__trajectory.json"
)
data2 = read_json_cartesian_pose(
    "crowndata-evaluation/example_data/data/droid_00000001/trajectories/cartesian_position__trajectory.json"
)
print(data.shape)
print(data2.shape)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    # Ensure both arrays have the same shape
    if a.shape != b.shape:
        raise ValueError("Arrays must have the same shape.")

    # Calculate the dot product of the two arrays
    dot_product = np.dot(a, b)

    # Calculate the magnitude (norm) of each array
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    # Return the cosine similarity
    return dot_product / (norm_a * norm_b)
