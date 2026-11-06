import math
import random

import pytest

from safejev.core import calibration, registry, taxonomy
from safejev.core.fusion import fuse
from safejev.core.result import Detection, ScanResult
from safejev.core.schema import Label, Record, RecordInput, Source, read_jsonl, write_jsonl


def _rec(**kw):
    base = dict(
        id="tsbench-train-000123-step4",
        source=Source("TS-Bench", "train", "v1", "?"),
        granularity="step", module="agent",
        input=RecordInput(user_goal="Find duplicate files in /documents",
                          proposed_action={"tool": "delete_file",
                                           "arguments": {"path": "/documents/report.pdf"}}),
        label=Label("ask", ["intent_mismatch", "destructive_action"], "high"),
        original_label={"raw": "controversial", "scheme": "TS-Bench"},
        meta={"risky_step_index": 4},
    )
    base.update(kw)
    return Record(**base)


def test_taxonomy_consistent():
    assert set(taxonomy.DEFAULT_SEVERITY) == set(taxonomy.ALL_TYPES)
    assert len(taxonomy.ALL_TYPES) == len(set(taxonomy.ALL_TYPES))
    assert taxonomy.max_severity(["low", "critical", "high"]) == "critical"
    assert taxonomy.max_severity([]) == "none"
    assert taxonomy.max_decision(["allow", "block", "ask"]) == "block"
    assert taxonomy.risk_slice(["injected_goal"]) == "external"
    assert taxonomy.risk_slice(["intent_mismatch"]) == "intrinsic"
    assert taxonomy.risk_slice(["injected_goal", "malformed_call"]) == "mixed"
    with pytest.raises(ValueError):
        taxonomy.severity_rank("huge")


def test_record_roundtrip(tmp_path):
    r = _rec().validate()
    p = tmp_path / "x.jsonl"
    assert write_jsonl([r, r], p) == 2
    back = list(read_jsonl(p))
    assert back[0] == r and len(back) == 2


@pytest.mark.parametrize("bad", [
    dict(module="nope"),
    dict(label=Label("allow", ["intent_mismatch"], "none")),
    dict(label=Label("block", [], "none")),
    dict(label=Label("ask", ["jailbreak"], "high")),          # guard type in agent record
    dict(input=RecordInput(user_goal="x")),                   # step without action
])
def test_record_validation_rejects(bad):
    with pytest.raises(ValueError):
        _rec(**bad).validate()


def test_fusion_picks_highest_severity():
    ds = [Detection("agent", "malformed_call", 0.99, "a", "low"),
          Detection("agent", "intent_mismatch", 0.7, "a", "high"),
          Detection("agent", "data_exfiltration", 0.2, "a", "critical")]     # not fired
    f = fuse(ds)
    assert f.risk_level == "high"
    assert [d.type for d in f.fired] == ["intent_mismatch", "malformed_call"]
    assert fuse([]).risk_level == "none" and fuse([]).top is None


def test_scanresult_json():
    r = ScanResult("ask", "high", [Detection("agent", "intent_mismatch", 0.912345, "a", "high")])
    d = r.to_dict()
    assert d["detections"][0]["score"] == 0.9123 and not r.allowed
    assert '"decision": "ask"' in r.to_json()


def test_temperature_recovers_scale():
    rng = random.Random(0)
    logits, labels = [], []
    for _ in range(2000):
        y = rng.randrange(3)
        true = [rng.gauss(0, 1) + (2.0 if k == y else 0.0) for k in range(3)]
        p = calibration.softmax(true)
        y = rng.choices(range(3), weights=p)[0]
        logits.append([3.0 * z for z in true])     # overconfident by x3
        labels.append(y)
    t = calibration.fit_temperature(logits, labels)
    assert 2.4 < t < 3.6
    before = calibration.multiclass_ece([calibration.softmax(z) for z in logits], labels)
    after = calibration.multiclass_ece([calibration.softmax(z, t) for z in logits], labels)
    assert after < before


def test_ece_and_bins():
    assert calibration.ece([0.9, 0.9], [1, 1]) == pytest.approx(0.1)
    assert calibration.ece([], []) == 0.0
    bins = calibration.reliability_bins([0.05, 0.95, 1.0], [0, 1, 1])
    assert [b[3] for b in bins] == [1, 2]
    assert calibration.sigmoid(-800) == 0.0 and math.isclose(calibration.sigmoid(0), 0.5)
    t = calibration.fit_binary_temperature([5.0, -5.0, 5.0, -5.0], [1, 0, 0, 1])
    assert t > 5


def test_registry():
    assert registry.parse_ref("safejev/agent-precheck@0.1.0") == ("agent-precheck", "0.1.0")
    assert registry.get("agent-tsbench").tier == "research"
    assert registry.defaults_for(["guard", "agent"]) == [
        "guard-injection", "guard-jailbreak", "agent-precheck"]
    with pytest.raises(KeyError):
        registry.get("nope")
    with pytest.raises(KeyError):
        registry.defaults_for(["toxicity"])
