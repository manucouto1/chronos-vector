"""Tests for the chronos_vector Python bindings.

Run with: cd crates/cvx-python && maturin develop && pytest tests/
"""
import pytest


@pytest.fixture
def cvx():
    """Import chronos_vector module."""
    import chronos_vector
    return chronos_vector


@pytest.fixture
def populated_index(cvx):
    """Create an index with 20 points for entity 1 (linear trajectory)."""
    index = cvx.TemporalIndex(m=16, ef_construction=100, ef_search=50)
    for i in range(20):
        ts = i * 1_000_000
        vec = [i * 0.1, (20 - i) * 0.1, 0.5]
        index.insert(entity_id=1, timestamp=ts, vector=vec)
    return index


# ─── TemporalIndex ────────────────────────────────────────────────

class TestTemporalIndex:
    def test_create_empty(self, cvx):
        index = cvx.TemporalIndex()
        assert len(index) == 0
        assert repr(index) == "TemporalIndex(len=0)"

    def test_insert_and_len(self, cvx):
        index = cvx.TemporalIndex()
        index.insert(entity_id=1, timestamp=1000, vector=[0.1, 0.2, 0.3])
        assert len(index) == 1
        index.insert(entity_id=1, timestamp=2000, vector=[0.4, 0.5, 0.6])
        assert len(index) == 2

    def test_search_basic(self, populated_index):
        results = populated_index.search(
            vector=[1.0, 1.0, 0.5], k=5
        )
        assert len(results) == 5
        for entity_id, timestamp, score in results:
            assert isinstance(entity_id, int)
            assert isinstance(timestamp, int)
            assert isinstance(score, float)

    def test_search_with_filter(self, populated_index):
        results = populated_index.search(
            vector=[1.0, 1.0, 0.5],
            k=20,
            filter_start=5_000_000,
            filter_end=10_000_000,
        )
        for _, ts, _ in results:
            assert 5_000_000 <= ts <= 10_000_000

    def test_trajectory(self, populated_index):
        traj = populated_index.trajectory(entity_id=1)
        assert len(traj) == 20
        # Verify ordering
        timestamps = [ts for ts, _ in traj]
        assert timestamps == sorted(timestamps)
        # Verify vectors have correct dim
        for _, vec in traj:
            assert len(vec) == 3

    def test_trajectory_with_range(self, populated_index):
        traj = populated_index.trajectory(entity_id=1, start=5_000_000, end=10_000_000)
        assert len(traj) == 6  # 5M, 6M, 7M, 8M, 9M, 10M
        for ts, _ in traj:
            assert 5_000_000 <= ts <= 10_000_000

    def test_trajectory_unknown_entity(self, populated_index):
        traj = populated_index.trajectory(entity_id=999)
        assert len(traj) == 0

    def test_predict_linear(self, populated_index):
        predicted, method = populated_index.predict(
            entity_id=1, target_timestamp=25_000_000
        )
        assert len(predicted) == 3
        assert method == "linear"
        assert all(isinstance(v, float) for v in predicted)

    def test_predict_insufficient_data(self, cvx):
        index = cvx.TemporalIndex()
        index.insert(entity_id=1, timestamp=1000, vector=[0.1, 0.2])
        with pytest.raises(ValueError, match="insufficient data"):
            index.predict(entity_id=1, target_timestamp=2000)

    def test_has_neural_ode_false_by_default(self, cvx):
        index = cvx.TemporalIndex()
        assert index.has_neural_ode() is False


# ─── Standalone functions ─────────────────────────────────────────

class TestVelocity:
    def test_basic(self, cvx):
        trajectory = [
            (0, [0.0, 0.0]),
            (1000, [1.0, 2.0]),
            (2000, [2.0, 4.0]),
        ]
        vel = cvx.velocity(trajectory, timestamp=1000)
        assert len(vel) == 2
        assert all(isinstance(v, float) for v in vel)

    def test_insufficient_data(self, cvx):
        with pytest.raises(ValueError):
            cvx.velocity([(0, [1.0])], timestamp=0)


class TestDrift:
    def test_basic(self, cvx):
        l2, cosine, top_dims = cvx.drift(
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], top_n=2
        )
        assert l2 > 0.0
        assert isinstance(cosine, float)
        assert len(top_dims) <= 2
        # First dim should have largest change
        assert top_dims[0][0] == 0  # dimension index
        assert top_dims[0][1] == pytest.approx(1.0)

    def test_identical_vectors(self, cvx):
        l2, _, _ = cvx.drift([1.0, 2.0], [1.0, 2.0])
        assert l2 == pytest.approx(0.0, abs=1e-6)


class TestChangepoints:
    def test_stationary(self, cvx):
        trajectory = [(i * 1000, [1.0, 2.0]) for i in range(100)]
        cps = cvx.detect_changepoints(entity_id=1, trajectory=trajectory)
        assert len(cps) == 0

    def test_with_change(self, cvx):
        trajectory = []
        for i in range(50):
            trajectory.append((i * 1000, [0.0, 0.0]))
        for i in range(50, 100):
            trajectory.append((i * 1000, [10.0, 10.0]))
        cps = cvx.detect_changepoints(entity_id=1, trajectory=trajectory)
        assert len(cps) >= 1
        # Changepoint should be near t=50000
        ts_list = [ts for ts, _ in cps]
        assert any(abs(ts - 50000) < 10000 for ts in ts_list)


class TestTemporalFeatures:
    def test_basic(self, cvx):
        trajectory = [(i * 1000, [float(i) * 0.1, float(i) * 0.2]) for i in range(20)]
        features = cvx.temporal_features(trajectory)
        assert isinstance(features, list)
        assert len(features) > 0
        assert all(isinstance(f, float) for f in features)


class TestHurstExponent:
    def test_basic(self, cvx):
        # Random walk-like trajectory
        import random
        random.seed(42)
        trajectory = []
        x, y = 0.0, 0.0
        for i in range(200):
            x += random.gauss(0, 0.1)
            y += random.gauss(0, 0.1)
            trajectory.append((i * 1000, [x, y]))

        h = cvx.hurst_exponent(trajectory)
        assert isinstance(h, float)
        assert 0.0 < h < 1.0


class TestPredict:
    def test_standalone(self, cvx):
        trajectory = [
            (0, [0.0, 0.0]),
            (1000, [1.0, 2.0]),
            (2000, [2.0, 4.0]),
        ]
        predicted = cvx.predict(trajectory, target_timestamp=3000)
        assert len(predicted) == 2
        # Linear extrapolation: [3.0, 6.0]
        assert predicted[0] == pytest.approx(3.0, abs=0.1)
        assert predicted[1] == pytest.approx(6.0, abs=0.1)
