import argparse
import copy
import contextlib
import io
from pathlib import Path
import tempfile
import unittest

from memtracker_nccl.run_metadata import (
    K3_MODEL_PROFILE, add_metadata_arguments, apply_dataset_metadata, compare_metadata,
    metadata_from_args, modeled_metadata, resolve_metadata,
)
from memtracker_nccl.report_io import read_report, write_report

ROOT = Path(__file__).resolve().parents[1]


class RunMetadataTests(unittest.TestCase):
    def test_one_cli_option_per_component_and_old_options_rejected(self):
        parser = argparse.ArgumentParser()
        add_metadata_arguments(parser)
        options = {option for action in parser._actions for option in action.option_strings}
        self.assertEqual(options - {'-h', '--help', '--run-metadata', '--runtime-profile'},
                         {'--training-code', '--torchao', '--pytorch', '--cuda', '--nccl', '--container'})
        for option in ('--training-code-commit', '--training-code-base-commit', '--training-code-patch-sha256',
                       '--container-identifier', '--container-digest', '--nccl-version', '--nccl-build'):
            with self.subTest(option=option), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parser.parse_args([option, 'value'])

    def test_single_nccl_input_clears_fallback_build(self):
        profile = {'schema_version': 1, 'name': 'old', 'values': {'nccl': '2.30.7+cuda13.3'}}
        r = resolve_metadata(profile=profile, layers=[('run', {'nccl': '2.31.0'})])
        self.assertEqual(r['values']['nccl_version'], '2.31.0')
        self.assertIsNone(r['values']['nccl_build'])
        r = resolve_metadata(profile=profile, layers=[('run', {'nccl': None})])
        self.assertIsNone(r['values']['nccl_version'])
        self.assertIsNone(r['values']['nccl_build'])

    def test_single_container_input_parses_digest_and_clears_it_on_override(self):
        pinned = 'registry/image@sha256:' + 'a'*64
        r = resolve_metadata(layers=[('run', {'container': pinned})])
        self.assertEqual(r['values']['container_digest'], 'sha256:' + 'a'*64)
        profile = {'schema_version': 1, 'name': 'old', 'values': {'container': pinned}}
        r = resolve_metadata(profile=profile, layers=[('run', {'container': 'other.sqsh'})])
        self.assertEqual(r['values']['container_identifier'], 'other.sqsh')
        self.assertIsNone(r['values']['container_digest'])

    def test_saved_fallback_null_details_cannot_erase_explicit_build(self):
        data = {'experiments': [{'job': 1, 'run_metadata': resolve_metadata(profile=K3_MODEL_PROFILE)}]}
        r = apply_dataset_metadata(data, manifest={'schema_version': 1, 'shared': {'nccl': '2.30.7+cuda13.3'}})
        self.assertEqual(r['experiments'][0]['run_metadata']['values']['nccl_build'], '2.30.7+cuda13.3')

    def test_tracker_target_identity_is_separate_from_simulator_and_immutable(self):
        from memtracker_nccl.tracker import ExtendedMemTracker
        values = {"pytorch": "target-version"}
        tracker = ExtendedMemTracker(run_metadata=values, runtime_profile=K3_MODEL_PROFILE)
        values["pytorch"] = "mutated"
        result = tracker.report()
        self.assertEqual(result["run_metadata"]["values"]["pytorch_version"], "target-version")
        self.assertEqual(result["torch_version"], result["simulator_environment"]["pytorch_version"])
        result["run_metadata"]["values"]["pytorch_version"] = "mutated-again"
        self.assertEqual(tracker.report()["run_metadata"]["values"]["pytorch_version"], "target-version")

    def test_explicit_values_override_defaults_and_null_masks_default(self):
        result = resolve_metadata(profile=K3_MODEL_PROFILE, layers=[("run", {
            "pytorch": "different-runtime", "training_code": None})])
        self.assertEqual(result["values"]["pytorch_version"], "different-runtime")
        self.assertIsNone(result["values"]["training_code_commit"])
        self.assertEqual(result["origins"]["training_code_commit"], "run")
        self.assertNotIn("training_code_commit", result["assumed_fields"])
        self.assertIn("cuda_version", result["assumed_fields"])
        self.assertEqual(len(result["overrides"]), 2)

    def test_no_automatic_host_or_container_version_inference(self):
        r = resolve_metadata(layers=[("run", {"container": "some-cu130.sqsh"})])
        self.assertIsNone(r["values"]["pytorch_version"])
        self.assertIsNone(r["values"]["cuda_version"])
        self.assertEqual(r["assumed_fields"], [])

    def test_base_commit_never_becomes_exact_commit(self):
        r = resolve_metadata(layers=[("run", {"training_code": "a"*40})])
        self.assertIsNone(r["values"]["training_code_commit"])
        comparison = compare_metadata(r, modeled_metadata({"source_commit": "b"*40}))
        self.assertEqual(comparison["status"], "incomplete")
        self.assertEqual(comparison["differences"], [])

    def test_base_mismatch_is_not_hidden_by_matching_container(self):
        a = resolve_metadata(layers=[("run", {"training_code": "a"*40, "container": "same.sqsh"})])
        b = resolve_metadata(layers=[("model", {"training_code": "b"*40, "container": "same.sqsh"})])
        self.assertEqual(compare_metadata(a, b)["status"], "mismatch")

    def test_different_paths_do_not_prove_different_contents(self):
        a = resolve_metadata(layers=[("run", {**K3_MODEL_PROFILE["values"], "container": "a.sqsh"})])
        b = resolve_metadata(layers=[("model", {**K3_MODEL_PROFILE["values"], "container": "b.sqsh"})])
        self.assertEqual(compare_metadata(a, b)["status"], "incomplete")
        self.assertFalse(compare_metadata(a, b)["identity_verified"])

    def test_fallback_is_not_declared_or_verified_match(self):
        assumed = resolve_metadata(profile=K3_MODEL_PROFILE)
        model = modeled_metadata({"source_commit": K3_MODEL_PROFILE["values"]["training_code"]})
        self.assertEqual(compare_metadata(assumed, model)["status"], "incomplete")
        self.assertIn("pytorch_version", compare_metadata(assumed, model)["assumed_fields"])

    def test_manifest_overrides_shared_but_embedded_run_wins(self):
        data = {"experiments": [{"job": 1}, {"job": 2, "run_metadata":
            resolve_metadata(layers=[("dataset:run", {"cuda": "14.0"})])}]}
        before = copy.deepcopy(data)
        manifest = {"schema_version": 1, "shared": {"cuda": "13.0"},
                    "runs": {"1": {"cuda": "12.0"}, "2": {"cuda": "12.0"}}}
        result = apply_dataset_metadata(data, profile=K3_MODEL_PROFILE, manifest=manifest)
        self.assertEqual([e["run_metadata"]["values"]["cuda_version"] for e in result["experiments"]], ["12.0", "14.0"])
        self.assertEqual(data, before)
        self.assertEqual(apply_dataset_metadata(result), result)
        with self.assertRaises(ValueError):
            apply_dataset_metadata(data, manifest={"schema_version": 1, "runs": {"3": {}}})

    def test_saved_fallback_cannot_override_new_explicit_manifest(self):
        data = {"experiments": [{"job": 1, "run_metadata": resolve_metadata(profile=K3_MODEL_PROFILE)}]}
        manifest = {"schema_version": 1, "shared": {"cuda": "12.0"}}
        result = apply_dataset_metadata(data, manifest=manifest)
        metadata = result["experiments"][0]["run_metadata"]
        self.assertEqual(metadata["values"]["cuda_version"], "12.0")
        self.assertNotIn("cuda_version", metadata["assumed_fields"])
        self.assertEqual(apply_dataset_metadata(result, manifest=manifest), result)

    def test_invalid_or_misspelled_fields_rejected(self):
        for values in ({"cuda_version": "13"}, {"training_code": "abcd"},
                       {"nccl": 23007}, {"pytorch": " "},
                       {"container_digest": "a"*64}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                resolve_metadata(layers=[("run", values)])

    def test_cli_overrides_file_and_named_profile(self):
        with tempfile.TemporaryDirectory() as directory:
            metadata = Path(directory)/"run.json"
            profile = Path(directory)/"profile.json"
            write_report(metadata, {"cuda": "12.0", "nccl": "2.30.7+cuda13.3"})
            write_report(profile, K3_MODEL_PROFILE)
            parser = argparse.ArgumentParser()
            add_metadata_arguments(parser)
            args = parser.parse_args(["--run-metadata", str(metadata), "--runtime-profile", str(profile),
                                      "--cuda", "13.0"])
            r = metadata_from_args(args)
            self.assertEqual(r["values"]["cuda_version"], "13.0")
            self.assertEqual(r["origins"]["cuda_version"], "cli")
            self.assertEqual(r["values"]["nccl_build"], "2.30.7+cuda13.3")
            self.assertEqual(r["origins"]["pytorch_version"], "default_profile:k3-b4d5-cu130")

    def test_supplied_historical_identity_preserved_for_all_nine_runs(self):
        data = read_report(ROOT/"experiments/gb200_historical_observations.json")
        manifest = read_report(ROOT/"experiments/gb200_run_metadata.json")
        result = apply_dataset_metadata(data, manifest=manifest)
        self.assertEqual(len(result["experiments"]), 9)
        for e in result["experiments"]:
            v = e["run_metadata"]["values"]
            self.assertEqual(v["training_code_base_commit"], "53a45ee31d260bcaacaa319dae29080c68675a49")
            self.assertIsNone(v["training_code_commit"])
            self.assertEqual(v["nccl_build"], "2.30.7+cuda13.3")
            self.assertEqual(v["cuda_version"], "13.0")


if __name__ == "__main__":
    unittest.main()
