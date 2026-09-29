"""Small CPU fixtures for the independent prediction-artifact audit."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from evaluate_mbhm import classification_metrics, corpus_summary, text_metrics
from verify_prediction_coverage import check_pairs, check_heldout, check_corpus


class CoverageTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="bearllm-coverage-test-")
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name)

    def write(self, name, value):
        (self.path / name).write_text(json.dumps(value) + "\n")

    def jsonl(self, name, rows):
        (self.path / name).write_text("".join(json.dumps(row) + "\n" for row in rows))

    def test_pairs_require_probabilities_all_draws_and_exact_confusion(self):
        path = self.path / "vibration_predictions.jsonl"
        expected = {(0, i): (0, 0, "test", i) for i in range(3)}
        metadata = {0: (7, 0, "fixture")}
        records = [{"file_id": 0, "reference_index": i, "ref_id": 0, "label": 0, "split": "test",
                    "pair_index": i, "condition_id": 7, "source": "fixture",
                    **{name: {"prediction": 0, "probabilities": [1.] + [0.] * 9}
                       for name in ("pre_lora_adapter", "final_lora_adapter")}} for i in range(3)]
        metrics = {"complete": True, "smoke_test_only": False, "count_pairs": 3, "count_unique_queries": 1,
                   "metrics": {name: {"all": classification_metrics([(0, 0)] * 3)}
                               for name in ("pre_lora_adapter", "final_lora_adapter")}}
        self.jsonl(path.name, records); self.write("vibration_metrics.json", metrics)
        self.assertEqual(check_pairs(path, expected, metadata)[1]["pairs"], 3)
        bad = copy.deepcopy(records); bad[0]["final_lora_adapter"]["probabilities"][0] = float("nan")
        self.jsonl(path.name, bad)
        with self.assertRaises(ValueError): check_pairs(path, expected, metadata)
        self.jsonl(path.name, records[:2])
        with self.assertRaises(ValueError): check_pairs(path, expected, metadata)
        self.jsonl(path.name, records)
        metrics["metrics"]["final_lora_adapter"]["all"]["confusion_matrix"][0][-1] = 1
        self.write("vibration_metrics.json", metrics)
        with self.assertRaises(ValueError): check_pairs(path, expected, metadata)

    def test_heldout_requires_text_parser_and_aggregate_agreement(self):
        path = self.path / "heldout_predictions.jsonl"
        expected = {5: (0, 1, 12)}
        metadata = {5: (7, 1, "fixture")}
        row = {"file_id": 5, "ref_id": 0, "label": 1, "pair_index": 12, "reference_index": 0,
               "split": "test", "condition_id": 7, "source": "fixture", "prediction": "unrecognized",
               "parsed_prediction": -1, "adapter_prediction": 0, "generated_tokens": 0, "truncated": False}
        metrics = {"complete": True, "smoke_test_only": False, "count": 1,
                   "truncated_outputs": 0, "mean_generated_tokens": 0.,
                   "metrics": {name: {group: classification_metrics([(1, predicted)]) for group in ("all", "source/fixture")}
                               for name, predicted in (("generated_classification", -1), ("adapter_classification", 0))}}
        self.jsonl(path.name, [row]); self.write("heldout_metrics.json", metrics)
        self.assertEqual(check_heldout(path, expected, metadata)["count"], 1)
        for mutate in (lambda value: value.pop("prediction"), lambda value: value.update(parsed_prediction=0),
                       lambda value: value.update(source="wrong")):
            bad = copy.deepcopy(row); mutate(bad); self.jsonl(path.name, [bad])
            with self.assertRaises((ValueError, KeyError)): check_heldout(path, expected, metadata)
        self.jsonl(path.name, [row])
        metrics["metrics"]["generated_classification"]["all"]["accuracy"] = 1.
        self.write("heldout_metrics.json", metrics)
        with self.assertRaises(ValueError): check_heldout(path, expected, metadata)

    def test_corpus_requires_outputs_overlap_and_aggregate_agreement(self):
        path = self.path / "corpus_predictions.jsonl"
        corpus = {i: {"id": i, "task_id": i, "instruction": "fixture", "label_id": 1,
                      "vib_id": 5, "ref_id": 0, "response": response, "condition_id": 7}
                  for i, response in ((0, "yes"), (1, "Minor Inner Ring Fault"))}
        metadata = {5: (7, 1, "fixture")}
        records = [{**entry, "source": "fixture", "prediction": "unrecognized", "parsed_prediction": -1,
                    "adapter_prediction": 0, "generated_tokens": 1, "truncated": False,
                    "reference_nll_sum": 1., "reference_token_count": 1,
                    **text_metrics("unrecognized", entry["response"])} for entry in corpus.values()]
        metrics = corpus_summary(records, 2, False)
        self.jsonl(path.name, records); self.write("corpus_metrics.json", metrics)
        self.assertEqual(check_corpus(path, corpus, metadata)["count"], 2)
        bad = copy.deepcopy(records); bad[0].pop("prediction"); self.jsonl(path.name, bad)
        with self.assertRaises(KeyError): check_corpus(path, corpus, metadata)
        bad = copy.deepcopy(records); bad[0].update(prediction="yes", parsed_prediction=1); self.jsonl(path.name, bad)
        with self.assertRaises(ValueError): check_corpus(path, corpus, metadata)
        self.jsonl(path.name, records)
        metrics["metrics"]["all"]["reference_perplexity"] = 100.
        self.write("corpus_metrics.json", metrics)
        with self.assertRaises(ValueError): check_corpus(path, corpus, metadata)


if __name__ == "__main__":
    unittest.main()
