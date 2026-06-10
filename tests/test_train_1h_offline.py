"""End-to-end offline test of train_1h_model.main() with a stubbed builder kit.

Plants a directional-only signal (sign predictable from one feature, magnitude
pure noise) so the classifier family should win, then verifies the exported
predict.pkl reloads and returns a finite float.
"""
import sys, types, importlib.util, os, pickle
import numpy as np
import pandas as pd

rng = np.random.default_rng(7)
N_BARS = 128
N_ROWS = 3000

def make_window_df(n_rows, seed_offset=0):
    """Synthetic kit-style dataframe: normalized OHLCV windows + target where
    the NEXT return's SIGN follows the last bar's return sign with p=0.56
    (mean reversion), but magnitude is independent noise."""
    r = np.random.default_rng(99 + seed_offset).normal(0, 0.004, (n_rows, N_BARS))
    data = {}
    C = np.exp(np.cumsum(r, axis=1)); C = C / C[:, -1:]
    H = C * (1 + np.abs(np.random.default_rng(1).normal(0, 0.0008, (n_rows, N_BARS))))
    L = C * (1 - np.abs(np.random.default_rng(2).normal(0, 0.0008, (n_rows, N_BARS))))
    V = np.abs(np.random.default_rng(3).lognormal(0, 0.5, (n_rows, N_BARS))); V = V / V[:, -1:]
    for i in range(N_BARS):
        for f, M in [("open", C), ("high", H), ("low", L), ("close", C), ("volume", V)]:
            data[f"feature_{f}_{i}"] = M[:, i]
    df = pd.DataFrame(data)
    df["open_time"] = pd.date_range("2025-06-01", periods=n_rows, freq="1h", tz="UTC")
    last_ret = np.log(C[:, -1]) - np.log(C[:, -2])
    mag = np.abs(np.random.default_rng(4).normal(0, 0.004, n_rows))
    flip = np.random.default_rng(5).random(n_rows) < 0.56          # 56% mean-revert
    sign = np.where(flip, -np.sign(last_ret), np.sign(last_ret))
    df["target"] = sign * mag
    return df

DF = make_window_df(N_ROWS)

class FakeWorkflow:
    def __init__(self, **kw): pass
    def backfill(self, start=None): pass
    def get_full_feature_target_dataframe(self, start_date=None):
        return DF.set_index("open_time")
    def get_live_features(self, ticker):
        return make_window_df(1, seed_offset=42).drop(columns=["target"]).set_index("open_time")

class FakeEvaluator:
    def evaluate(self, y_true, y_pred):
        da = float(np.mean(np.sign(y_pred) == np.sign(y_true)))
        return {"num_passed": 0, "score": 0.0, "grade": "n/a", "da": da}
    def print_report(self, report, detailed=True):
        print(f"  [fake evaluator] DA={report['da']:.4f}")

kit = types.ModuleType("allora_forge_builder_kit")
kit.AlloraMLWorkflow = FakeWorkflow
kit.PerformanceEvaluator = FakeEvaluator
sys.modules["allora_forge_builder_kit"] = kit

os.environ["DAYS_OF_HISTORY"] = "200"
os.environ["PREDICT_PKL"] = "/tmp/test_predict.pkl"
os.environ["ALLORA_API_KEY"] = "UP-fake-for-test"

# shrink the grid so the test runs fast
_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "train_1h_model", os.path.join(_here, "..", "scripts", "train_1h_model.py"))
t = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(t)
t.LEARNING_RATES = [0.05]
t.MAX_DEPTHS = [3]
t.NUM_LEAVES = [15]
t.N_ESTIMATORS_CHECKPOINTS = [100]
t.N_ESTIMATORS_MAX = 100

t.main()

predict = pickle.load(open("/tmp/test_predict.pkl", "rb"))
val = predict(12345)
assert np.isfinite(val), "predict.pkl returned non-finite"
print(f"\nreloaded predict.pkl -> {val:+.6f}")
print("END-TO-END OFFLINE TEST PASS")
