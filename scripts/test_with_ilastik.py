from pathlib import Path
from typing import Optional

import typer

from bioimageio.core import load_resource_description
from bioimageio.core.resource_io.nodes import Model, ResourceDescription
from bioimageio.core.resource_tests import test_model
from bioimageio.spec.shared import yaml


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
        test_name = "reproduce test outputs with ilastik <todo version>"
        try:
            rd: Optional[ResourceDescription] = load_resource_description(rdf_path)
        except Exception as e:
            error = f"Unable to load rdf: {e}"
            write_summary(
                dict(name=test_name, error=error, status="failed"), dist / resource_id / version_id / f"test_summary.yaml"
            )
            continue

        if rd.type != "model":
            write_summary(
                dict(status="skipped"), dist / resource_id / version_id / f"test_summary.yaml"
            continue

        assert isinstance(rd, Model)
        for weight_format in ["onnx", "torchscript", "pytorch_state_dict"]:
            if weight_format in rd.weights:
                summary = test_model(rd, weight_format=weight_format)
                summary["name"] = f"{test_name} using {weight_format} weights"
                write_summary(summary, dist / resource_id / version_id / f"test_summary_{weight_format}.yaml")


if __name__ == "__main__":
    typer.run(main)
