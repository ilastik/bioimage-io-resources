import traceback
from pathlib import Path
from typing import List

import typer
from bioimageio.core import __version__ as bioimageio_core_version
from bioimageio.core.common import TestSummary
from bioimageio.core.resource_tests import test_model
from bioimageio.spec import __version__ as bioimageio_spec_version
from bioimageio.spec.shared import yaml


def write_summaries(
    p: Path,
    summaries: List[TestSummary],
):
    p.parent.mkdir(parents=True, exist_ok=True)
    yaml.dump(summaries, p)


def run_tests(
    rdf_dir: Path, resource_id: str, version_id: str, summaries_dir: Path, postfix: str
):

    summary_defaults = {
        "bioimageio_spec_version": bioimageio_spec_version,
        "bioimageio_core_version": bioimageio_core_version,
        "warnings": {},
        "nested_errors": None,
    }

    for rdf_path in rdf_dir.glob(f"{resource_id}/{version_id}/rdf.yaml"):
        test_name = f"reproduce test outputs with ilastik {postfix}"
        error = None
        status = None

        try:
            rdf = yaml.load(rdf_path)
        except Exception as e:
            error = f"Unable to load rdf: {e}"
            status = "failed"
            rdf = {}

        rd_id = rdf.get("id")
        if rd_id is None or not isinstance(rd_id, str):
            print(
                f"::warning file=scripts/test_with_ilastik.py,line=37,endline=41,title=Invalid RDF::"
                f"Missing/invalid 'id' in rdf {str(rdf_path.relative_to(rdf_dir).parent)}"
            )
            continue

        if rdf.get("type") != "model":
            status = "skipped"
            error = "not a model RDF"

        weight_formats = list(rdf.get("weights", []))
        if not isinstance(weight_formats, list) or not weight_formats:
            status = "failed"
            error = f"Missing/invalid weight formats for {rd_id}"

        if status:
            # write single test summary
            summaries = [
                TestSummary(
                    dict(
                        name=test_name,
                        status=status,
                        error=error,
                        source_name=str(rdf_path),
                        **summary_defaults,
                    )
                )
            ]
            write_summaries(
                summaries_dir / rd_id / f"test_summary_{postfix}.yaml", summaries
            )
            continue

        # write test summary for each weight format
        for weight_format in weight_formats:
            try:
                summaries = test_model(rdf_path, weight_format=weight_format)
            except Exception as e:
                summary = dict(
                    error=str(e), traceback=traceback.format_tb(e.__traceback__)
                )

                summary["name"] = f"{test_name} using {weight_format} weights"
                summary["status"] = "failed"
                summaries = [TestSummary(**summary, **summary_defaults)]

            write_summaries(
                summaries_dir / rd_id / f"test_summary_{weight_format}_{postfix}.yaml",
                summaries,
            )


def main(
    dist: Path,
    resource_id: str,
    version_id: str = "**",
    rdf_dir: Path = Path(__file__).parent / "../bioimageio-gh-pages/rdfs",
    postfix: str = "",
):
    """preliminary ilastik check

    only checks if test outputs are reproduced for onnx, torchscript, or pytorch_state_dict weights.

    """
    summaries_dir = dist / "test_summaries"
    summaries_dir.mkdir(parents=True, exist_ok=True)
    run_tests(rdf_dir, resource_id, version_id, summaries_dir, postfix)


if __name__ == "__main__":
    typer.run(main)
