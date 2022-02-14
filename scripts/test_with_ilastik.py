from pathlib import Path

import typer

from bioimageio.core import load_resource_description
from bioimageio.core.resource_io.nodes import ResourceDescription
from bioimageio.core.resource_tests import test_model
from bioimageio.spec.shared import yaml

try:
    from typing import Optional, get_args
except ImportError:
    from typing_extensions import get_args  # type: ignore


def write_summary(s: dict, p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)
    yaml.dump(s, p)


def main(
    dist: Path, resource_id: str, version_id: str = "**", rdf_dir: Path = Path(__file__).parent / "../bioimageio-gh-pages/rdfs"
):
    """ preliminary ilastik check

    only checks if test outputs are reproduced and onnx, torchscript, or pytorch_state_dict weights are available.

    """
    dist.mkdir(parents=True, exist_ok=True)
    for rdf_path in rdf_dir.glob(f"{resource_id}/{version_id}/rdf.yaml"):
        try:
            rd: Optional[ResourceDescription] = load_resource_description(rdf_path)
        except Exception as e:
            error = f"Unable to load rdf: {e}"
        else:
            error = None

        test_name = "reproduce test outputs with ilastik <todo version>"
        if error is None:
            for weight_format in ["onnx", "torchscript", "pytorch_state_dict"]:
                if error is not None:
                    if weight_format in rd.weights:
                        summary = test_model(rd, weight_format=weight_format)
                        summary["name"] = f"{test_name} using {weight_format} weights"
                        write_summary(summary, dist / resource_id / version_id / f"test_summary_{weight_format}.yaml")
        else:
            write_summary(
                dict(name=test_name, error=error, status="failed"), dist / resource_id / version_id / f"test_summary.yaml"
            )


if __name__ == "__main__":
    typer.run(main)
