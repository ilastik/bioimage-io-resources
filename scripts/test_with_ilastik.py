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


def run_tests(rdf_dir: Path, resource_id: str, version_id: str, summaries_dir: Path, postfix: str):
    summary_defaults = {
        "bioimageio_spec_version": bioimageio_spec_version,
        "bioimageio_core_version": bioimageio_core_version,
        "warnings": {},
    }

    for rdf_path in rdf_dir.glob(f"{resource_id}/{version_id}/rdf.yaml"):
        test_name = f"Reproduce test outputs with ilastik {postfix}"
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
            write_summaries(summaries_dir / rd_id / f"test_summary_{postfix}.yaml", summaries)
            continue

        # produce test summaries for each weight format
        summaries_per_weight_format = {}
        for weight_format in weight_formats:
            try:
                summaries_weight_format = test_model(rdf_path, weight_format=weight_format)
            except Exception as e:
                summaries_weight_format = [
                    TestSummary(
                        name=test_name,
                        status="failed",
                        error=str(e),
                        traceback=traceback.format_tb(e.__traceback__),
                        **summary_defaults,
                    )
                ]

            summaries_per_weight_format[weight_format] = summaries_weight_format

        assert summaries_per_weight_format
        # merge test weight format summaries to one ilastik summary
        passed_reproduced_summaries = []
        failed_reproduced_summaries = []
        other_summaries = []
        seen_tests = set()
        for wf, s in summaries_per_weight_format.items():
            for ss in s:
                is_other = ss["name"].lower() != "reproduce test outputs from test inputs"
                if is_other:
                    # filter out redundant, wf independent tests summaries
                    if str(ss) in seen_tests:
                        continue

                    seen_tests.add(str(ss))

                if is_other:
                    ss["name"] = f"{ss['name']} ({wf})"
                    other_summaries.append(ss)
                    continue

                ss["name"] = f"{test_name} ({wf})"
                if ss["status"] == "passed":
                    passed_reproduced_summaries.append(ss)
                else:
                    failed_reproduced_summaries.append(ss)

        chosen_summaries = passed_reproduced_summaries or failed_reproduced_summaries or other_summaries
        # todo: save failed tests also if other pass. we filter them out for now to not falsely conclude that a model
        #  cannot be run in ilastik only because some of the tests failed (while others passed)
        write_summaries(
            summaries_dir / rd_id / f"test_summary_{postfix}.yaml",
            chosen_summaries,
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
