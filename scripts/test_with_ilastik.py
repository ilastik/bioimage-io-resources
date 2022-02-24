from pathlib import Path

import typer

from bioimageio.core.resource_tests import test_model
from bioimageio.spec.shared import yaml


def write_summary(s: dict, p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)
    yaml.dump(s, p)


def write_test_summaries(rdf_dir: Path, resource_id: str, version_id: str, summaries_dir: Path):
    for rdf_path in rdf_dir.glob(f"{resource_id}/{version_id}/rdf.yaml"):
        test_name = "reproduce test outputs with ilastik <todo version> (draft)"
        error = None
        status = None
        reason = None
        try:
            rdf = yaml.load(rdf_path)
        except Exception as e:
            error = f"Unable to load rdf: {e}"
            status = "failed"
            rdf = {}

        rd_id = rdf.get("id")
        if rd_id is None or not isinstance(rd_id, str):
            print(
                f"::warning file=scripts/test_with_ilastik.py,line=40,endline=44,title=Invalid RDF::"
                f"Missing/invalid 'id' in rdf {str(rdf_path.relative_to(rdf_dir).parent)}"
            )
            continue

        if rdf.get("type") != "model":
            status = "skipped"
            reason = "not a model RDF"

        if status:
            # write single test summary
            write_summary(
                dict(status=status, error=error, reason="error" if error else reason),
                summaries_dir / rd_id / f"test_summary.yaml",
            )
            continue

        # write test summary for each weight format
        for weight_format in ["onnx", "torchscript", "pytorch_state_dict"]:
            summary = test_model(rdf_path, weight_format=weight_format)
            summary["name"] = f"{test_name} using {weight_format} weights"
            write_summary(summary, summaries_dir / rd_id / f"test_summary_{weight_format}.yaml")


def main(
    dist: Path,
    resource_id: str,
    version_id: str = "**",
    rdf_dir: Path = Path(__file__).parent / "../bioimageio-gh-pages/rdfs",
):
    """ preliminary ilastik check

    only checks if test outputs are reproduced for onnx, torchscript, or pytorch_state_dict weights.

    """
    summaries_dir = dist / "test_summaries"
    summaries_dir.mkdir(parents=True, exist_ok=True)
    write_test_summaries(rdf_dir, resource_id, version_id, summaries_dir)


if __name__ == "__main__":
    typer.run(main)
