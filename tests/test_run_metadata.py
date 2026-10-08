import argparse
import copy
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
    def test_tracker_target_identity_is_separate_from_simulator_and_immutable(self):
        from memtracker_nccl.tracker import ExtendedMemTracker
        values = {"pytorch_version": "target-version"}
        tracker = ExtendedMemTracker(run_metadata=values, runtime_profile=K3_MODEL_PROFILE)
        values["pytorch_version"] = "mutated"
        result = tracker.report()
        self.assertEqual(result["run_metadata"]["values"]["pytorch_version"], "target-version")
        self.assertEqual(result["torch_version"], result["simulator_environment"]["pytorch_version"])
        result["run_metadata"]["values"]["pytorch_version"] = "mutated-again"
        self.assertEqual(tracker.report()["run_metadata"]["values"]["pytorch_version"], "target-version")

    def test_explicit_values_override_defaults_and_null_masks_default(self):
        result = resolve_metadata(profile=K3_MODEL_PROFILE, layers=[("run", {
            "pytorch_version": "different-runtime", "training_code_commit": None})])
        self.assertEqual(result["values"]["pytorch_version"], "different-runtime")
        self.assertIsNone(result["values"]["training_code_commit"])
        self.assertEqual(result["origins"]["training_code_commit"], "run")
        self.assertNotIn("training_code_commit", result["assumed_fields"])
        self.assertIn("cuda_version", result["assumed_fields"])
        self.assertEqual(len(result["overrides"]), 2)

    def test_no_automatic_host_or_container_version_inference(self):
        r = resolve_metadata(layers=[("run", {"container_identifier": "some-cu130.sqsh"})])
        self.assertIsNone(r["values"]["pytorch_version"])
        self.assertIsNone(r["values"]["cuda_version"])
        self.assertEqual(r["assumed_fields"], [])

    def test_base_commit_never_becomes_exact_commit(self):
        r = resolve_metadata(layers=[("run", {"training_code_base_commit": "a"*40})])
        self.assertIsNone(r["values"]["training_code_commit"])
        comparison = compare_metadata(r, modeled_metadata({"source_commit": "b"*40}))
        self.assertEqual(comparison["status"], "incomplete")
        self.assertEqual(comparison["differences"], [])

    def test_exact_mismatch_is_not_hidden_by_matching_container(self):
        a = resolve_metadata(layers=[("run", {"training_code_commit": "a"*40, "container_identifier": "same.sqsh"})])
        b = resolve_metadata(layers=[("model", {"training_code_commit": "b"*40, "container_identifier": "same.sqsh"})])
        self.assertEqual(compare_metadata(a, b)["status"], "mismatch")

    def test_different_paths_do_not_prove_different_contents(self):
        a = resolve_metadata(layers=[("run", {**K3_MODEL_PROFILE["values"], "container_identifier": "a.sqsh"})])
        b = resolve_metadata(layers=[("model", {**K3_MODEL_PROFILE["values"], "container_identifier": "b.sqsh"})])
        self.assertEqual(compare_metadata(a, b)["status"], "declared_match")
        self.assertFalse(compare_metadata(a, b)["identity_verified"])

    def test_fallback_is_not_declared_or_verified_match(self):
        assumed = resolve_metadata(profile=K3_MODEL_PROFILE)
        model = modeled_metadata({"source_commit": K3_MODEL_PROFILE["values"]["training_code_commit"]})
        self.assertEqual(compare_metadata(assumed, model)["status"], "assumed_match")

    def test_manifest_overrides_shared_but_embedded_run_wins(self):
        data = {"experiments": [{"job": 1}, {"job": 2, "run_metadata":
            resolve_metadata(layers=[("dataset:run", {"cuda_version": "14.0"})])}]}
        before = copy.deepcopy(data)
        manifest = {"schema_version": 1, "shared": {"cuda_version": "13.0"},
                    "runs": {"1": {"cuda_version": "12.0"}, "2": {"cuda_version": "12.0"}}}
        result = apply_dataset_metadata(data, profile=K3_MODEL_PROFILE, manifest=manifest)
        self.assertEqual([e["run_metadata"]["values"]["cuda_version"] for e in result["experiments"]], ["12.0", "14.0"])
        self.assertEqual(data, before)
        self.assertEqual(apply_dataset_metadata(result), result)
        with self.assertRaises(ValueError):
            apply_dataset_metadata(data, manifest={"schema_version": 1, "runs": {"3": {}}})

    def test_saved_fallback_cannot_override_new_explicit_manifest(self):
        data = {"experiments": [{"job": 1, "run_metadata": resolve_metadata(profile=K3_MODEL_PROFILE)}]}
        manifest = {"schema_version": 1, "shared": {"cuda_version": "12.0"}}
        result = apply_dataset_metadata(data, manifest=manifest)
        metadata = result["experiments"][0]["run_metadata"]
        self.assertEqual(metadata["values"]["cuda_version"], "12.0")
        self.assertNotIn("cuda_version", metadata["assumed_fields"])
        self.assertEqual(apply_dataset_metadata(result, manifest=manifest), result)

    def test_invalid_or_misspelled_fields_rejected(self):
        for values in ({"cuda": "13"}, {"training_code_commit": "abcd"},
                       {"nccl_version": 23007}, {"pytorch_version": " "},
                       {"container_digest": "a"*64}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                resolve_metadata(layers=[("run", values)])

    def test_cli_overrides_file_and_named_profile(self):
        with tempfile.TemporaryDirectory() as directory:
            metadata = Path(directory)/"run.json"
            profile = Path(directory)/"profile.json"
            write_report(metadata, {"cuda_version": "12.0", "nccl_build": "2.30.7+cuda13.3"})
            write_report(profile, K3_MODEL_PROFILE)
            parser = argparse.ArgumentParser()
            add_metadata_arguments(parser)
            args = parser.parse_args(["--run-metadata", str(metadata), "--runtime-profile", str(profile),
                                      "--cuda-version", "13.0"])
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
