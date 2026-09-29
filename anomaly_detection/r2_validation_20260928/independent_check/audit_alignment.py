"""Read-only, independent reconstruction of saved R2 neural experiments.

Does not import experiment training functions. Writes only in independent_check.
Re-fits training-only scalers/readouts to verify their saved states; does not
train or replace any experimental student or classifier checkpoint.
"""
import os
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "2"
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
GROUPS = [[0], [1, 2, 3], [7, 8, 9], [4, 5, 6]]
torch.set_num_threads(2)


def digest(path):
    hh = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for part in iter(lambda: fh.read(1024 * 1024), b""):
            hh.update(part)
    return hh.hexdigest()


def weights(ids):
    ids = np.asarray(ids)
    unique, counts = np.unique(ids, return_counts=True)
    lookup = dict(zip(unique, counts))
    return np.array([len(ids) / len(unique) / lookup[x] for x in ids])


def close(a, b, tol=1e-9, context=""):
    error = float(np.max(np.abs(np.asarray(a) - np.asarray(b))))
    if not np.allclose(a, b, atol=tol, rtol=tol):
        raise AssertionError(f"{context}: absolute error {error}")
    return error


class ReloadEncoder(nn.Module):
    def __init__(self, mean, scale):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(128, 128), nn.LayerNorm(128), nn.GELU(),
                                 nn.Dropout(.1), nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 128))
        self.register_buffer("contact_mean", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("contact_scale", torch.tensor(scale, dtype=torch.float32))

    def forward(self, x):
        return torch.relu(self.contact_mean + self.contact_scale * self.net(x))


class ReloadDirect(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(128, 128), nn.LayerNorm(128), nn.GELU(),
                                 nn.Dropout(.1), nn.Linear(128, 64), nn.GELU(),
                                 nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 4))

    def forward(self, x):
        return self.net(x)


def metrics(y, prediction):
    y, prediction = np.asarray(y), np.asarray(prediction)
    return dict(accuracy=float(accuracy_score(y, prediction)),
                macro_f1=float(f1_score(y, prediction, labels=range(4), average="macro", zero_division=0)),
                correct=int(np.sum(y == prediction)), n=len(y))


def audit():
    c = np.load(ROOT / "contact/contact_features.npz", allow_pickle=False)
    r = np.load(ROOT / "radar/features.npz", allow_pickle=False)
    h = c["embedding"].astype(np.float32)
    yy = c["physical_targets"].astype(float)
    cy, crpm, csid = c["labels"], c["rpm"], c["session_id"].astype(str)
    y, rpm, rid, sid = r["labels"], r["rpm"], r["recording_id"].astype(str), r["contact_id"].astype(str)
    outputs = []
    skipped = []
    for dd in sorted((ROOT / "experiments").iterdir()):
        if not dd.is_dir() or not (dd / "run_protocol.json").exists():
            continue
        if not (dd / "verification.json").exists() or not json.loads((dd / "verification.json").read_text()).get("completed"):
            skipped.append(dd.name)
            continue
        outputs.append(dd)
    reload_rows, fit_rows, metric_rows, pool_rows, source_rows = [], [], [], [], []
    for dd in outputs:
        config = json.loads((dd / "run_protocol.json").read_text())
        for source, expected in config["source_hashes"].items():
            current = digest(source)
            assert current == expected, f"changed source {source}"
            source_rows.append(dict(experiment=dd.name, path=source, sha256=current))
        assert digest(ROOT / "experiments/run_alignment.py") == config["program_sha256"]
        feature = config["feature"]
        x = r[feature].astype(np.float32)
        splits = json.loads((dd / "splits.json").read_text())
        original_metrics = pd.read_csv(dd / "metrics.csv")
        record_csv = pd.read_csv(dd / "recording_predictions.csv")
        assert len(splits) == 3
        pooled = {}
        for split in splits:
            fold = split["fold"]
            hold = int(fold.split("_")[-1])
            rt, ct = rpm != hold, crpm != hold
            test_ids = np.flatnonzero(~rt)
            train_s, test_s = sorted(set(csid[ct])), sorted(set(csid[~ct]))
            assert len(train_s) == 8 and len(test_s) == 4 and not set(train_s) & set(test_s)
            assert split["train_contact_sessions"] == train_s and split["test_contact_sessions"] == test_s
            assert split["train_recordings"] == sorted(set(rid[rt]))
            assert split["test_recordings"] == sorted(set(rid[~rt]))
            assert not set(split["train_recordings"]) & set(split["test_recordings"])
            assert set(sid[rt]) == set(train_s) and set(sid[~rt]) == set(test_s)
            assert split["train_radar_windows"] == rt.sum() and split["test_radar_windows"] == (~rt).sum()
            assert split["train_contact_packets"] == ct.sum() and split["test_contact_packets"] == (~ct).sum()

            target = np.load(dd / "targets" / f"{fold}.npz", allow_pickle=False)
            assert np.array_equal(target["train_sessions"], train_s) and np.array_equal(target["test_sessions"], test_s)
            csc = StandardScaler().fit(h[ct], sample_weight=weights(csid[ct]))
            rsc = StandardScaler().fit(x[rt], sample_weight=weights(rid[rt]))
            errors = {}
            for key, expected in [("contact_mean", csc.mean_), ("contact_scale", csc.scale_),
                                  ("radar_mean", rsc.mean_), ("radar_scale", rsc.scale_)]:
                errors[key] = close(target[key], expected, context=key)
            hz = csc.transform(h).astype(np.float32)
            targets = np.stack([hz[csid == ss].mean(0) for ss in train_s])
            labels_s = np.array([cy[csid == ss][0] for ss in train_s])
            prototypes = np.stack([targets[labels_s == cl].mean(0) for cl in range(4)])
            errors["targets"] = close(target["targets"], targets)
            errors["class_prototypes"] = close(target["class_prototypes"], prototypes)
            ys = StandardScaler().fit(yy[ct], sample_weight=weights(csid[ct]))
            readout = Ridge(alpha=10, solver="svd").fit(hz[ct], ys.transform(yy[ct]), sample_weight=weights(csid[ct]))
            for key, expected in [("physical_mean", ys.mean_), ("physical_scale", ys.scale_),
                                  ("readout_coef", readout.coef_), ("readout_intercept", readout.intercept_)]:
                errors[key] = close(target[key], expected, context=key)
            if config["head"] == "archived":
                errors["head_weight"] = close(target["head_weight"], c["retained_weight10"], tol=0)
                errors["head_bias"] = close(target["head_bias"], c["retained_bias10"], tol=0)
            else:
                hd = np.load(ROOT / "contact/diagnostic_heads" / f"{fold}.npz", allow_pickle=False)
                assert not bool(hd["accepted_for_source_retention"])
                assert np.array_equal(hd["train_sessions"], train_s) and np.array_equal(hd["test_sessions"], test_s)
                errors["head_weight"] = close(target["head_weight"], hd["weight4"], tol=0)
                errors["head_bias"] = close(target["head_bias"], hd["bias4"], tol=0)
                # Independently verify that these are the documented contact-only train-RPM heads.
                hd_scaler = StandardScaler().fit(h[ct].astype(float), sample_weight=weights(csid[ct]))
                hd_model = LogisticRegression(C=1., max_iter=3000, solver="lbfgs", random_state=42)
                hd_model.fit(hd_scaler.transform(h[ct].astype(float)), cy[ct], sample_weight=weights(csid[ct]))
                ww = hd_model.coef_ / hd_scaler.scale_[None, :]
                bb = hd_model.intercept_ - ww @ hd_scaler.mean_
                errors["diagnostic_train_only_weight"] = close(target["head_weight"], ww.astype(np.float32), tol=1e-5)
                errors["diagnostic_train_only_bias"] = close(target["head_bias"], bb.astype(np.float32), tol=1e-5)
                assert "NOT certified" in config["head_certification"]
            fit_rows.append(dict(experiment=dd.name, fold=fold, all_train_only_fits_match=True,
                                 max_saved_parameter_error=max(errors.values()), component_errors=errors))

            xx = torch.from_numpy(rsc.transform(x)).float()
            w = torch.from_numpy(target["head_weight"]).float()
            b = torch.from_numpy(target["head_bias"]).float()
            for pp in sorted((dd / "predictions").glob(f"{fold}__*.npz")):
                saved = np.load(pp, allow_pickle=False)
                _, method, seed_name = pp.stem.split("__")
                seed = int(seed_name.removeprefix("seed"))
                assert np.array_equal(saved["row_index"], test_ids)
                assert np.array_equal(saved["labels"], y[test_ids])
                assert np.array_equal(saved["recording_id"], rid[test_ids])
                assert np.array_equal(saved["contact_id"], sid[test_ids])
                saved_p = saved["probabilities4"]
                assert saved_p.shape == (len(test_ids), 4) and np.isfinite(saved_p).all()
                close(saved_p.sum(1), np.ones(len(test_ids)), tol=1e-6)
                ck = torch.load(dd / "models" / f"{pp.stem}.pt", map_location="cpu", weights_only=False)
                assert ck["feature"] == feature and ck["head"] == config["head"] and ck["fold"] == fold and ck["method"] == method
                close(ck["radar_mean"], rsc.mean_)
                close(ck["radar_scale"], rsc.scale_)
                direct = method == "direct_MLP"
                model = ReloadDirect() if direct else ReloadEncoder(csc.mean_, csc.scale_)
                model.load_state_dict(ck["state_dict"], strict=True)
                model.eval()
                with torch.inference_mode():
                    hidden = model(xx)
                    logits = hidden if direct else hidden @ w.T + b
                    if logits.shape[-1] == 10:
                        logits = torch.stack([torch.logsumexp(logits[:, gg], dim=1) for gg in GROUPS], dim=1)
                    probability = logits.softmax(-1).numpy()
                max_prob = close(probability[test_ids], saved_p, tol=2e-5, context=pp.stem)
                max_hidden = None if direct else close(hidden.numpy()[test_ids], saved["hidden"], tol=2e-5, context="hidden")

                rec_rows = record_csv[(record_csv.fold == fold) & (record_csv.method == method) & (record_csv.seed == seed)]
                assert len(rec_rows) == 144 and rec_rows.recording_id.nunique() == 144
                rec_truth, rec_pred, rec_ids = [], [], []
                rec_error = 0.
                for row in rec_rows.itertuples(index=False):
                    ix = rid == row.recording_id
                    assert ix.any() and set(y[ix]) == {row.label} and set(sid[ix]) == {row.contact_id}
                    assert set(rpm[ix]) == {row.rpm} and set(r["distance_cm"][ix]) == {row.distance_cm}
                    assert row.n_windows == ix.sum()
                    assert row.role == ("test" if row.rpm == hold else "train")
                    actual = probability[ix].mean(0)
                    rec_error = max(rec_error, close(actual, [getattr(row, f"p{k}") for k in range(4)], tol=2e-5))
                    assert int(actual.argmax()) == row.prediction
                    if row.role == "test":
                        rec_truth.append(row.label); rec_pred.append(row.prediction); rec_ids.append(row.recording_id)
                assert len(rec_ids) == 48
                metrics_to_check = [("window", y[test_ids], saved_p.argmax(1)),
                                    ("recording", np.array(rec_truth), np.array(rec_pred))]
                for cm in [20, 40, 80]:
                    mask = r["distance_cm"][test_ids] == cm
                    metrics_to_check.append((f"window_{cm}cm", y[test_ids][mask], saved_p[mask].argmax(1)))
                for level, truth, prediction in metrics_to_check:
                    result = metrics(truth, prediction)
                    matching = original_metrics[(original_metrics.fold == fold) & (original_metrics.method == method)
                                                & (original_metrics.seed == seed) & (original_metrics.level == level)]
                    assert len(matching) == 1
                    for key, value in result.items():
                        close(float(matching.iloc[0][key]), value, tol=1e-12, context=f"metric {key}")
                    metric_rows.append(dict(experiment=dd.name, fold=fold, method=method, seed=seed, level=level, **result))
                    if level in ("window", "recording"):
                        pooled.setdefault((method, seed, level), []).append((np.array(truth), np.array(prediction)))
                reload_rows.append(dict(experiment=dd.name, fold=fold, method=method, seed=seed,
                                        probability_max_abs_error=max_prob, hidden_max_abs_error=max_hidden,
                                        recording_probability_max_abs_error=rec_error, test_rows_exact=True,
                                        test_windows=len(test_ids), test_recordings=len(rec_ids)))
            print(f"{dd.name} {fold} verified", flush=True)
        for (method, seed, level), parts in pooled.items():
            assert len(parts) == 3
            truth = np.concatenate([p[0] for p in parts])
            pred = np.concatenate([p[1] for p in parts])
            assert len(truth) == (2158 if level == "window" else 144)
            pool_rows.append(dict(experiment=dd.name, method=method, seed=seed, level=level, **metrics(truth, pred)))

    pd.DataFrame(reload_rows).to_csv(HERE / "checkpoint_forward_checks.csv", index=False)
    pd.DataFrame(metric_rows).to_csv(HERE / "recomputed_metrics.csv", index=False)
    pooled_df = pd.DataFrame(pool_rows)
    pooled_df.to_csv(HERE / "pooled_oof_by_seed.csv", index=False)
    pooled_df.groupby(["experiment", "method", "level"])[["accuracy", "macro_f1"]].agg(["mean", "min", "max"]).to_csv(HERE / "pooled_oof_seed_summary.csv")
    (HERE / "train_fit_reconstruction.json").write_text(json.dumps(fit_rows, ensure_ascii=False, indent=2) + "\n")
    result = dict(status="passed", completed_experiments=[d.name for d in outputs], skipped_incomplete=skipped,
                  checkpoint_forward_checks=len(reload_rows), recomputed_metric_rows=len(metric_rows),
                  folds_with_independently_reconstructed_train_only_fits=len(fit_rows),
                  max_probability_abs_error=max(row["probability_max_abs_error"] for row in reload_rows),
                  max_recording_probability_abs_error=max(row["recording_probability_max_abs_error"] for row in reload_rows),
                  every_saved_prediction_contains_only_expected_heldout_rpm_rows=True,
                  recorded_train_predictions_are_explicitly_separate_from_test=True,
                  no_shared_contact_sessions_between_train_and_test=True,
                  archived_actual_ten_class_weights_match_source=True,
                  diagnostic_heads_reconstructed_from_training_contact_only=True,
                  diagnostic_source_retention_certified=False,
                  optimizer_head_freezing_evidence="training code optimizes model.parameters() only; w,b created from arrays with requires_grad=False; saved target heads match immutable source arrays",
                  loss_interpretation="full is predeclared CE + mean-feature MSE + 0.1 class-positive contrast + 0.5 temperature-2 KD; this audit verifies execution, not novelty or superiority",
                  scope="No stronger independent-bearing/motor/environment or severity claim; only 12 shared contact sessions; 3 seeds do not multiply physical samples",
                  audited_source_hashes=source_rows, auditor_sha256=digest(__file__))
    (HERE / "verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "audited_source_hashes"}, indent=2), flush=True)


if __name__ == "__main__":
    audit()
