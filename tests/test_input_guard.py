"""A failed slide must stop a monitor, never decide its result.

Before the guard, one NaN gave Tier 1 its smallest possible p-value and left Tier 2 unable to flag.
"""

import numpy as np
import pytest

from wsidrift.monitor.divergence import Tier1Monitor
from wsidrift.monitor.embedding import Tier2Monitor

rng = np.random.default_rng(0)
X, Y = rng.normal(size=(60, 8)), rng.normal(size=(40, 8))
G, H = np.arange(60) // 2, np.arange(40) // 2


def spoiled(a, value=np.nan):
    b = a.copy()
    b[0, 0] = value
    return b


CALLS = {
    "tier1 reference": lambda v: Tier1Monitor(n_permutations=9).fit(spoiled(X, v), G),
    "tier1 deployment": lambda v: Tier1Monitor(n_permutations=9).fit(X, G).score(spoiled(Y, v), H),
    "tier1 no groups": lambda v: Tier1Monitor(n_permutations=9).fit(X).score(spoiled(Y, v)),
    "tier1 fixed thresholds": lambda v: Tier1Monitor(mmd_threshold=.1, wass_threshold=.1).fit(X).score(spoiled(Y, v)),
    "tier2 reference": lambda v: Tier2Monitor().fit(spoiled(X, v), groups=G),
    "tier2 deployment": lambda v: Tier2Monitor().fit(X, groups=G).score(spoiled(Y, v), groups=H),
    "tier2 reference logits": lambda v: Tier2Monitor().fit(X, spoiled(X[:, :2], v), groups=G),
    "tier2 deployment logits": lambda v: Tier2Monitor().fit(X, X[:, :2], groups=G).score(Y, spoiled(Y[:, :2], v), groups=H),
}


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("call", CALLS.values(), ids=CALLS.keys())
def test_non_finite_input_is_refused(call, value):
    with pytest.raises(ValueError, match="NaN or infinite"):
        call(value)


def test_empty_input_is_refused_in_words():
    with pytest.raises(ValueError, match="empty"):
        Tier1Monitor(n_permutations=9).fit(X).score(Y[:0])


def test_the_message_says_which_rows():
    with pytest.raises(ValueError, match="1 of 40 .*row 0"):
        Tier1Monitor(n_permutations=9).fit(X).score(spoiled(Y))


def test_clean_input_is_untouched():
    r = Tier1Monitor(n_permutations=49).fit(X, G).score(Y, H)
    assert np.isfinite(r.mmd_squared) and r.p_mmd > 1 / 50
    assert np.isfinite(Tier2Monitor().fit(X, groups=G).score(Y, groups=H).mahalanobis_mean)


def test_cli_stops_without_a_traceback(tmp_path):
    from wsidrift.cli import main
    a, b = tmp_path / "r.npy", tmp_path / "d.npy"
    np.save(a, X)
    np.save(b, spoiled(Y))
    with pytest.raises(SystemExit) as e:
        main(["score", "--reference", str(a), "--deployment", str(b), "--permutations", "9"])
    assert "NaN or infinite" in str(e.value.code)
