"""
Serialización de dicts a JSON tolerante a tipos no nativos (np.ndarray,
funciones, clases) — usado por el orquestador de preprocesamiento
(`dataset_description.json`, `preprocessing_pipeline.json`) y por
`PreprocessedDataset` para leerlos de vuelta.
"""
import json
import os
from pathlib import Path

def save_dict_as_json(path_to_save, dictionary, file_name="dict.json"):
    os.makedirs(path_to_save, exist_ok=True)
    full_path = os.path.join(path_to_save, file_name)
    with open(full_path, "w") as file:
        json.dump(dictionary, file, default=custom_serializer, indent=4)


def load_json_file(path):
    with open(path, "r") as file:
        return json.load(file)


def custom_serializer(obj):
    import numpy as np

    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if callable(obj):
        return obj.__name__ if hasattr(obj, "__name__") else str(obj)
    elif isinstance(obj, type):
        return obj.__module__ + "." + obj.__name__
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError("Object of type %s is not JSON serializable" % type(obj))
