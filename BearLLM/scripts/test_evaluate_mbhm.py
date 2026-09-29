"""Small correctness checks for evaluation protocol, parsing and batch injection."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch
from torch import nn

from evaluate_mbhm import (classification_metrics, classifier_logits, embed_batch, make_fallback_protocol, make_protocol,
                           parse_binary, parse_class, read_resume, text_metrics)


class EvaluationTests(unittest.TestCase):
    def test_protocol_groups_and_missing_reference(self):
        rows = [{"file_id": index, "condition_id": 1, "label": index % 10, "source": "fixture"} for index in range(12)]
        rows.append({"file_id": 12, "condition_id": 2, "label": 3, "source": "fixture"})
        pairs, exclusions = make_protocol(rows, 42, 3)
        self.assertEqual(len(pairs), 36)
        self.assertEqual(len(exclusions), 1)
        self.assertEqual([pairs[i * 3]["split"] for i in range(12)], ["train"] * 7 + ["val", "val", "test", "train", "train"])
        self.assertTrue(all(pair["ref_id"] in (0, 10) for pair in pairs))
        self.assertEqual((pairs, exclusions), make_protocol(rows, 42, 3))

    def test_unparseable_counts_as_error(self):
        report = classification_metrics([(0, 0), (1, -1), (1, 1)], classes=2)
        self.assertEqual(report["accuracy"], 2 / 3)
        self.assertEqual(report["invalid_predictions"], 1)
        self.assertEqual(report["confusion_matrix"], [[1, 0, 0], [0, 1, 1]])

    def test_fallback_keeps_sources_separate(self):
        rows = [{"file_id": 0, "condition_id": 0, "label": 0, "source": "a"},
                {"file_id": 1, "condition_id": 1, "label": 1, "source": "a"},
                {"file_id": 2, "condition_id": 2, "label": 0, "source": "b"},
                {"file_id": 3, "condition_id": 3, "label": 1, "source": "b"}]
        pairs, excluded = make_fallback_protocol(rows, 42, 2)
        self.assertEqual([(row["file_id"], row["ref_id"]) for row in pairs], [(1, 0), (1, 0), (3, 2), (3, 2)])
        self.assertEqual(len(excluded), 2)
        self.assertTrue(all(row["split"] == "excluded_from_upstream" for row in pairs))

    def test_parser_and_overlap(self):
        self.assertEqual(parse_class("**Minor Outer Ring Fault**. Not Fault-Free."), 7)
        self.assertEqual(parse_class("unclear"), -1)
        self.assertEqual(parse_binary("**Yes**, a fault exists"), 1)
        self.assertEqual(parse_binary("no"), 0)
        self.assertEqual(text_metrics("a b a", "a a b")["rouge_l_f1"], 2 / 3)
        self.assertEqual(text_metrics("Fault-Free", "fault free")["normalized_exact_match"], 1)

    def test_nonfinite_model_logits_are_rejected(self):
        layer = SimpleNamespace(linear1=nn.Linear(1, 1), relu=nn.ReLU(), linear2=nn.Linear(1, 10))
        with torch.no_grad():
            layer.linear1.weight.fill_(float("nan"))
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            classifier_logits(SimpleNamespace(alignment_layer=layer), torch.ones(1, 1))

    def test_resume_recovers_partial_line(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.jsonl"
            path.write_bytes(b'{"id":1}\n{"id":')
            self.assertEqual(read_resume(path, "id"), {1: {"id": 1}})
            self.assertEqual(path.read_text(), '{"id":1}\n')

    def test_resume_preserves_valid_line_without_newline(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.jsonl"
            path.write_bytes(b'{"id":1}')
            self.assertEqual(read_resume(path, "id"), {1: {"id": 1}})
            self.assertEqual(path.read_text(), '{"id":1}\n')

    def test_in_memory_embedding_preserves_batch_rows(self):
        from src.fine_tuning import signal_token_id

        class FakeModel:
            def __init__(self):
                self.embedding = nn.Embedding(10, 3)

            def get_input_embeddings(self):
                return self.embedding

        class FakeAdapter(nn.Module):
            def forward(self, signals):
                return signals[:, :1, :1].expand(-1, 5, 3)

        model = FakeModel()
        sequences = [torch.tensor([1] + [signal_token_id] * 5 + [2]),
                     torch.tensor([3, 4] + [signal_token_id] * 5 + [5, 6])]
        signals = torch.tensor([[[11.0]], [[22.0]]])
        ids, attention, embeddings = embed_batch(model, FakeAdapter(), sequences, signals, 0)
        self.assertEqual(attention.sum(-1).tolist(), [7, 9])
        self.assertTrue(torch.all(embeddings[0][ids[0] == signal_token_id] == 11))
        self.assertTrue(torch.all(embeddings[1][ids[1] == signal_token_id] == 22))
        self.assertTrue(torch.equal(embeddings[1, 0], model.embedding(torch.tensor(3))))


if __name__ == "__main__":
    unittest.main()
