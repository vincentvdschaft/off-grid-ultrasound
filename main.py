import os
import sys
import shutil
from pathlib import Path
import argparse
import yaml
import jax
from infer import (
    infer_from_file,
    info_writer,
    create_week_folder,
    create_date_folder,
    create_unique_dir,
)
from jaxus import use_dark_style, log

# Get the location of the current file
current_file = Path(__file__).resolve()
cache_dir = current_file.parent / "cache" / "jax"
cache_dir.mkdir(parents=True, exist_ok=True)
cache_dir = str(cache_dir)

# Set the cache directory for JAX
jax.config.update("jax_compilation_cache_dir", cache_dir)

parser = argparse.ArgumentParser()
parser.add_argument(
    "config",
    nargs="?",
    type=str,
    help="Path to the config file",
    default="config/infer_debug.yaml",
)

parser.add_argument(
    "--gpus", type=int, nargs="+", help="GPU devices to use", default=[0]
)

parser.add_argument("--seed", type=int, help="Which seed to use", default=0)
parser.add_argument("--output_dir", type=str, help="Which seed to use", default=None)
parser.add_argument(
    "--stamp", type=str, help="Stamp to add to the figures", default=None
)

args = parser.parse_args()

if args.gpus is not None:
    # Join the list of GPUs into a string
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, args.gpus))

path = Path(args.config)

if not path.exists():
    log.error(f"Config file {log.yellow(path)} does not exist.")
    sys.exit(1)

use_dark_style()

log.info(f"Reading config file from {log.yellow(path)}")

with open(path, "r", encoding="utf-8") as f:
    config_dict = yaml.safe_load(f)

info_writer.write_info("config_path", str(path))
info_writer.write_info("gpus", args.gpus)

run_name = "infer_run"


try:
    while True:
        try:
            if args.output_dir:
                working_dir = Path(args.output_dir)
                working_dir.mkdir(parents=True, exist_ok=True)
            else:
                working_dir = create_week_folder(Path("results"))
                working_dir = create_date_folder(working_dir)
                working_dir = create_unique_dir(
                    parent_directory=working_dir, name=run_name, prepend_date=True
                )

            # Copy the config file to the working directory
            config_path = working_dir / "run_config.yaml"
            shutil.copy(path, config_path)

            infer_from_file(
                config_dict=config_dict,
                working_dir=working_dir,
                seed=args.seed,
                stamp=args.stamp,
            )
            break
        except Exception as e:
            if "RESOURCE_EXHAUSTED" in str(e):
                log.error(
                    "Resource exhausted. Retrying with half the batch size and double "
                    "the gradient accumulation."
                )
                if config_dict["batch_size"] == 1:
                    log.error("Out of memory and batch_size is already at 1. Exiting.")
                    sys.exit(1)
                config_dict["batch_size"] = config_dict["batch_size"] // 2
                config_dict["gradient_accumulation"] = (
                    config_dict["gradient_accumulation"] * 2
                )
                info_writer.set_path(None)
                info_writer.write_info("adjusted_batch_size", config_dict["batch_size"])
                info_writer.write_info(
                    "adjusted_gradient_accumulation",
                    config_dict["gradient_accumulation"],
                )
                # Delete the working directory and its contents
                for file in working_dir.glob("*"):
                    file.unlink()
                working_dir.rmdir()
            else:
                raise e

except KeyboardInterrupt:
    log.warning("Program was interrupted by the user.")
    sys.exit(1)
