"""CPU-only boundary tests: synthetic files never enter the real experiment tree."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from evaluate_mbhm import classification_metrics
import summarize_reproduction as report


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


def classification(count, classes=10, prediction=-1):
    # Scaling one observation keeps the production metric schema without
    # allocating hundreds of thousands of synthetic prediction records.
    value = classification_metrics([(0, prediction)], classes)
    value["count"] *= count
    value["invalid_predictions"] *= count
    value["confusion_matrix"] = [[n * count for n in row] for row in value["confusion_matrix"]]
    for row in value["per_class"]:
        row["support"] *= count
    return value


def fcn_metrics(count):
    matrix = [[0] * 10 for _ in range(10)]
    matrix[0][1] = count  # A valid, deliberately zero-accuracy classifier.
    metric = {"n": count, "correct": 0, "accuracy": 0.0, "macro_f1_all_10_classes": 0.0,
              "balanced_accuracy_present_classes": 0.0, "false_alarm_rate": 1.0,
              "missed_alarm_rate": None, "confusion_matrix": matrix}
    return {"loss": 3.0, "by_dataset": {"MBHM": metric, "fixture": copy.deepcopy(metric)}}


def corpus_group(count, binary=0):
    value = {"count": count, "truncated_outputs": count, "mean_generated_tokens": 0.0,
             "reference_token_nll": 0.0, "reference_perplexity": 1.0,
             "text_overlap": {"normalized_exact_match": 0.0, "unigram_f1": 0.0, "rouge_l_f1": 0.0},
             "adapter_classification": classification(count)}
    if binary:
        value["generated_binary_detection"] = classification(binary, classes=2)
    if count > binary:
        value["generated_classification"] = classification(count - binary)
    return value


class ReporterTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="bearllm-report-test-")
        self.addCleanup(self.directory.cleanup)
        self.base = Path(self.directory.name)
        self.root, self.run = self.base / "repo", self.base / "run"
        self.data, self.qwen, self.official = self.root / "data", self.root / "qwen", self.root / "official"
        self.output = self.base / "output"
        for directory in (self.data, self.qwen, self.official, self.run / "pretrain/fcn", self.run / "finetune/weights"):
            directory.mkdir(parents=True)
        for directory, names in [(self.data, ("metadata.sqlite", "corpus.json")),
                                 (self.official, ("vibration_adapter.pth", "adapter_model.safetensors")),
                                 (self.run / "pretrain/fcn", ("feature_encoder.pth", "classifier.pth")),
                                 (self.run / "finetune/weights", ("vibration_adapter.pth", "adapter_model.safetensors"))]:
            for name in names:
                (directory / name).write_text(str(directory) + name)
        (self.root / ".env").write_text(f"MBHM_DATASET={self.data}\nQWEN_WEIGHTS={self.qwen}\nBEARLLM_WEIGHTS={self.official}\n")
        self.audit = {"scope": "complete", "checks": {str(i): True for i in range(15)},
                      "metadata": {"sample_count": 135516, "sources": {"fixture": {
                          "samples": 135516, "same_condition_healthy_reference_samples": 122792,
                          "missing_same_condition_healthy_reference_samples": 12724}}},
                      "corpus": {"source_counts": {"fixture": 600}}}
        write(self.root / "outputs/mbhm_audit.json", self.audit)
        write(self.root / "outputs/full/environment_ready.json", {"status": "passed"})
        self.metadata_hash, self.corpus_hash = (report.file_hash(self.data / n) for n in ("metadata.sqlite", "corpus.json"))
        self.fcn_hashes = {name: report.file_hash(self.run / "pretrain/fcn" / name)
                           for name in ("feature_encoder.pth", "classifier.pth")}
        config = {"seed": 42, "epochs": 50, "batch_size": 1024, "accumulation_steps": 4,
                  "learning_rate": 1e-4, "scheduler": "source-batch-loss", "data_dir": str(self.data),
                  "qwen_dir": str(self.qwen), "effective_batch_size": 1024, "disable_early_stop": False,
                  "metadata_sha256": self.metadata_hash, "corpus_sha256": self.corpus_hash, "stage": "pretrain",
                  "loader_protocol": {"train_shuffle": True, "validation_shuffle": True,
                                      "test_shuffle": True, "persistent_workers": False}}
        self.pretrain = {"stage": "pretrain", "requested_epochs": 50, "epochs_completed": 50,
                         "best_epoch": 1, "source_selected_epoch": 1,
                         "config": config, "protocol": {"seed": 42, "total_queries": 135516,
                         "eligible_queries": 122792, "pair_counts": report.PAIR_COUNTS, "query_counts": report.QUERY_COUNTS},
                         "exported_fcn_sha256": self.fcn_hashes,
                         **{key: fcn_metrics(36837) for key in ("last_checkpoint_test", "best_checkpoint_test", "source_selected_checkpoint_test")}}
        fine_config = {**config, "stage": "finetune", "batch_size": 1, "effective_batch_size": 4,
                       "fcn_dir": str(self.run / "pretrain/fcn"), "initial_fcn_sha256": self.fcn_hashes.copy(),
                       "loader_protocol": {"engine": "transformers.Trainer", "dataloader_num_workers": 0,
                                           "loss_normalization": "native Trainer model loss kwargs"}}
        native = {"finetune_engine": "transformers.Trainer", "model_accepts_loss_kwargs": True,
                  "loss_normalization": "native Trainer model loss kwargs"}
        fine_config.update(native)
        self.finetune = {"stage": "finetune", "requested_epochs": 50, "epochs_completed": 50,
                         "corpus_rows": 600, "global_step": 7500, "config": fine_config, **native,
                         "weights": str(self.run / "finetune/weights"),
                         "exported_weights_sha256": {name: report.file_hash(self.run / "finetune/weights" / name)
                                                     for name in ("vibration_adapter.pth", "adapter_model.safetensors")}}
        write(self.run / "pretrain/results.json", self.pretrain)
        write(self.run / "finetune/results.json", self.finetune)
        self.pretrain_epochs = [{"stage": "pretrain", "epoch": i, "train_pairs": 257865,
                                "train_loss": 3., "lr": 1e-4, "val": fcn_metrics(73674),
                                "source_checkpoint_selected": i == 1, "source_selected_epoch": 1} for i in range(1, 51)]
        self.finetune_epochs = [{"stage": "finetune", "epoch": i, "corpus_rows": 600,
                                "train_loss": 3., "lr": 1e-4, "global_step": 150 * i,
                                "unique_corpus_ids": 600} for i in range(1, 51)]
        self.write_epochs("pretrain", self.pretrain_epochs)
        self.write_epochs("finetune", self.finetune_epochs)
        write(self.run / "finetune/corpus_protocol.json", {"rows": 600, "total_optimizer_updates": 7500,
              "task_counts": {str(i): 150 for i in range(4)}, "label_counts": {str(i): 60 for i in range(10)}, **native})
        write(self.run / "finetune/trainer_arguments.json", {"per_device_train_batch_size": 1,
              "gradient_accumulation_steps": 4, "num_train_epochs": 50, "learning_rate": 1e-4,
              "lr_scheduler_type": "cosine", "seed": 42, "dataloader_num_workers": 0})
        for tag in ("official", "trained"):
            weights = self.official if tag == "official" else self.run / "finetune/weights"
            for suffix, stage, policy in (("evaluation", "all", "same-condition"),
                                          ("fallback", "vibration", "excluded-same-source-fallback"),
                                          ("heldout", "heldout-generation", "same-condition")):
                directory = self.run / (tag + "_" + suffix)
                write(directory / "run_config.json", {"stage": stage, "reference_policy": policy,
                      "seed": 42, "references": 3, "split": "test", "limit": None,
                      "checkpoint": str(weights), "qwen": str(self.qwen), "metadata_sha256": self.metadata_hash,
                      "corpus_sha256": self.corpus_hash, "vibration_adapter_sha256": report.file_hash(weights / "vibration_adapter.pth"),
                      "lora_sha256": report.file_hash(weights / "adapter_model.safetensors")})
                if suffix != "heldout":
                    count = 368376 if suffix == "evaluation" else 38172
                    groups = {"all": classification(count), "source/fixture": classification(count)}
                    if suffix == "evaluation":
                        groups.update({"split/" + split: classification(n) for split, n in report.PAIR_COUNTS.items()})
                    write(directory / "vibration_metrics.json", {"complete": True, "smoke_test_only": False,
                          "count_pairs": count, "count_unique_queries": count // 3, "reference_policy": policy,
                          "metrics": {name: copy.deepcopy(groups) for name in ("pre_lora_adapter", "final_lora_adapter")}})
                if suffix == "evaluation":
                    groups = {"all": corpus_group(600, 150), "source/fixture": corpus_group(600, 150)}
                    groups.update({"task/" + name: corpus_group(150, 150 if i == 0 else 0)
                                   for i, name in enumerate(report.CORPUS_TASKS)})
                    write(directory / "corpus_metrics.json", {"complete": True, "smoke_test_only": False,
                          "count": 600, "total": 600, "metrics": groups})
                if suffix == "heldout":
                    groups = {"all": classification(12279), "source/fixture": classification(12279)}
                    write(directory / "heldout_metrics.json", {"complete": True, "smoke_test_only": False,
                          "count": 12279, "truncated_outputs": 12279, "mean_generated_tokens": 0.,
                          "metrics": {name: copy.deepcopy(groups) for name in ("generated_classification", "adapter_classification")}})

    def write_epochs(self, stage, epochs):
        (self.run / stage / "epochs.jsonl").write_text("".join(json.dumps(row) + "\n" for row in epochs))

    def run_summary(self, complete):
        argv = ["summarize_reproduction.py", "--run-root", str(self.run), "--output", str(self.output), "--require-complete"]
        with patch.object(report, "ROOT", self.root), patch.object(report, "make_figures"), patch("sys.argv", argv), contextlib.redirect_stdout(io.StringIO()):
            if complete:
                report.main()
            else:
                with self.assertRaises(SystemExit):
                    report.main()
        summary = json.loads((self.output / "summary.json").read_text(), parse_constant=lambda value: self.fail(value))
        self.assertEqual(summary["complete_current_release_pipeline"], complete)
        return summary

    def test_zero_accuracy_unparseable_and_variable_lengths_are_valid(self):
        self.run_summary(True)

    def test_smoke_and_wrong_checkpoint_are_rejected(self):
        path = self.run / "official_evaluation/vibration_metrics.json"
        value = report.read(path); value["smoke_test_only"] = True; write(path, value)
        self.assertFalse(self.run_summary(False)["checks"]["official_all_supported_pairs"])
        value["smoke_test_only"] = False; write(path, value)
        path = self.run / "trained_evaluation/run_config.json"
        value = report.read(path); value["lora_sha256"] = "wrong"; write(path, value)
        self.assertFalse(self.run_summary(False)["checks"]["trained_checkpoint_and_protocol_provenance"])

    def test_nonfinite_metrics_never_certify_completion(self):
        path = self.run / "official_evaluation/corpus_metrics.json"
        value = report.read(path); value["metrics"]["all"]["reference_token_nll"] = float("nan"); write(path, value)
        summary = self.run_summary(False)
        self.assertFalse(summary["checks"]["official_all_corpus_rows"])
        self.assertIn("NaN", summary["validation_errors"]["official_all_corpus_rows"])
        self.assertNotIn("| nan |", (self.output / "RESULTS_zh.md").read_text())

    def test_classification_support_and_unique_queries_are_checked(self):
        path = self.run / "official_evaluation/vibration_metrics.json"
        original = report.read(path)
        for mutate in (lambda value: value.update(count_unique_queries=1),
                       lambda value: value["metrics"]["final_lora_adapter"].update(all=classification(1)),
                       lambda value: value["metrics"]["final_lora_adapter"]["all"].update(accuracy=1.1)):
            with self.subTest(mutate=mutate):
                value = copy.deepcopy(original); mutate(value); write(path, value)
                self.assertFalse(self.run_summary(False)["checks"]["official_all_supported_pairs"])

    def test_each_corpus_task_and_entire_heldout_set_are_required(self):
        path = self.run / "official_evaluation/corpus_metrics.json"
        value = report.read(path); value["metrics"]["task/maintenance"]["count"] = 149; write(path, value)
        self.assertFalse(self.run_summary(False)["checks"]["official_all_corpus_rows"])
        path = self.run / "official_heldout/heldout_metrics.json"
        value = report.read(path); value["count"] = 12278; write(path, value)
        self.assertFalse(self.run_summary(False)["checks"]["official_all_test_queries_generated"])

    def test_training_config_and_fcn_initialization_are_enforced(self):
        arguments = (self.run, self.data, self.qwen, self.metadata_hash, self.corpus_hash)
        for key, wrong in (("stage", "finetune"), ("seed", 123), ("scheduler", "validation-loss"),
                           ("batch_size", 32), ("accumulation_steps", 1), ("learning_rate", 1e-3)):
            with self.subTest(key=key):
                value = copy.deepcopy(self.pretrain); value["config"][key] = wrong
                self.assertIsNotNone(report.validation_error(report.validate_pretraining, value, *arguments))
        value = copy.deepcopy(self.finetune); value["config"]["initial_fcn_sha256"]["classifier.pth"] = "wrong"
        self.assertIsNotNone(report.validation_error(report.validate_finetuning, value, *arguments))
        value = copy.deepcopy(self.finetune); value["exported_weights_sha256"]["adapter_model.safetensors"] = "wrong"
        self.assertIsNotNone(report.validation_error(report.validate_finetuning, value, *arguments))
        value = copy.deepcopy(self.pretrain); value["config"]["loader_protocol"]["test_shuffle"] = False
        self.assertIsNotNone(report.validation_error(report.validate_pretraining, value, *arguments))
        value = copy.deepcopy(self.finetune); value["config"]["model_accepts_loss_kwargs"] = False
        self.assertIsNotNone(report.validation_error(report.validate_finetuning, value, *arguments))

    def test_incomplete_epoch_coverage_is_rejected(self):
        self.pretrain_epochs[5]["train_pairs"] -= 1
        self.write_epochs("pretrain", self.pretrain_epochs)
        self.finetune_epochs.pop()
        self.write_epochs("finetune", self.finetune_epochs)
        summary = self.run_summary(False)
        self.assertFalse(summary["checks"]["full_pretraining"])
        self.assertFalse(summary["checks"]["full_finetuning"])

    def test_logged_source_early_stop_is_accepted(self):
        self.pretrain.update(epochs_completed=3, best_epoch=1, source_selected_epoch=1)
        write(self.run / "pretrain/results.json", self.pretrain)
        epochs = self.pretrain_epochs[:3]; epochs[-1]["lr"] = 1e-8
        self.write_epochs("pretrain", epochs)
        self.run_summary(True)
        epochs[-1]["lr"] = 1e-4; self.write_epochs("pretrain", epochs)
        self.assertFalse(self.run_summary(False)["checks"]["full_pretraining"])

    def test_alarm_rates_use_true_class_denominators(self):
        metrics = classification_metrics([(0, 0), (0, 1), (1, 0), (1, 1), (2, 0)])
        rates = report.alarm_rates(metrics)
        self.assertEqual(rates["false_alarm_rate"], .5)
        self.assertEqual(rates["missed_alarm_rate"], 2 / 3)
        self.assertIsNone(report.alarm_rates(classification_metrics([(1, 0)]))["false_alarm_rate"])
        self.assertIsNone(report.alarm_rates(classification_metrics([(0, 0)]))["missed_alarm_rate"])


if __name__ == "__main__":
    unittest.main()
