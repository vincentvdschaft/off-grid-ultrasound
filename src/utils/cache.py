"""This module implements a decorator to cache the output of a function."""

import numpy as np
import hashlib
import functools
from pathlib import Path
import jax.numpy as jnp

from jaxus import log
import pickle
from itertools import chain
import inspect


def cache_outputs(dir_path: str):
    """Cache the output of a function to a directory under a hash of the arguments."""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):

            path = Path(dir_path)

            # Create the directory if it does not exist
            if not path.exists():
                path.mkdir(exist_ok=True, parents=True)

            # Create a string representation of the arguments
            args_str_components = []
            for arg in args:
                args_str_components.append(repr(arg))
                if isinstance(arg, (np.ndarray, jnp.ndarray)):
                    args_str_components.append(repr(arg.shape))
                    args_str_components.append(repr(arg.dtype))

            for key, value in kwargs.items():
                args_str_components.append(repr(key))
                args_str_components.append(repr(value))
                if isinstance(value, (np.ndarray, jnp.ndarray)):
                    args_str_components.append(repr(value.shape))
                    args_str_components.append(repr(value.dtype))

            arg_str = "_".join(args_str_components)

            # Add the source code of the function to the hash to change the hash if the
            # source code changes
            arg_str += inspect.getsource(func)

            # Create a hash of the string representation
            hash_obj = hashlib.sha256(arg_str.encode())
            hash_digest = hash_obj.hexdigest()

            # Formulate the file path
            filename = f"{func.__name__}_{hash_digest}.pkl"
            file_path = path / filename

            # Check if the file already exists
            if file_path.exists():
                # log.info(f"Loading cached result from {log.yellow(file_path)}")
                # Load and return the result
                with open(file_path, "rb") as file:
                    return pickle.load(file)

            # Call the function and get the result
            result = func(*args, **kwargs)

            # Save the result to a file
            with open(file_path, "wb") as file:
                pickle.dump(result, file)

            return result

        return wrapper

    return decorator
