"""CLI tests: the score command reads .npy and .csv without optional dependencies."""

import json

import numpy as np
import pytest

from wsidrift.cli import _load_features, main


def _write(tmp_path, name, X, header=False):
    path = tmp_path / name
    if name.endswith(".npy"):
        np.save(path, X)
    else:
        hdr = ",".join(f"f{i}" for i in range(X.shape[1])) if header else ""
        np.savetxt(path, X, delimiter=",", header=hdr, comments="")
    return str(path)


@pytest.mark.parametrize("header", [False, True])
def test_csv_loads_with_or_without_header(tmp_path, header):
    X = np.random.default_rng(0).standard_normal((12, 3))
    assert np.allclose(_load_features(_write(tmp_path, "x.csv", X, header)), X)


def test_single_column_csv_is_2d(tmp_path):
    X = np.arange(5.0).reshape(-1, 1)
    assert _load_features(_write(tmp_path, "x.csv", X)).shape == (5, 1)


def test_non_numeric_csv_exits_with_message(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("a,b\n1,x\n")
    with pytest.raises(SystemExit, match="could not read as numeric CSV"):
        main(["score", "--reference", str(path), "--deployment", str(path)])


@pytest.mark.parametrize("ext", [".npy", ".csv"])
def test_score_runs_end_to_end(tmp_path, ext, capsys):
    rng = np.random.default_rng(0)
    ref = _write(tmp_path, "ref" + ext, rng.standard_normal((60, 4)), header=True)
    dep = _write(tmp_path, "dep" + ext, rng.standard_normal((30, 4)) + 3, header=True)
    out = tmp_path / "result.json"
    main(["score", "--reference", ref, "--deployment", dep, "--permutations", "49", "--output", str(out)])
    assert json.loads(out.read_text())
    assert "saved" in capsys.readouterr().out


def test_score_with_groups(tmp_path):
    rng = np.random.default_rng(0)
    ref = _write(tmp_path, "ref.npy", rng.standard_normal((60, 4)))
    dep = _write(tmp_path, "dep.npy", rng.standard_normal((30, 4)) + 3)
    (tmp_path / "gr.txt").write_text("\n".join(f"s{i // 6}" for i in range(60)))
    (tmp_path / "gd.txt").write_text("\n".join(f"t{i // 6}" for i in range(30)))
    out = tmp_path / "r.json"
    main(["score", "--reference", ref, "--deployment", dep, "--permutations", "99",
          "--reference-groups", str(tmp_path / "gr.txt"), "--deployment-groups", str(tmp_path / "gd.txt"),
          "--output", str(out)])
    assert json.loads(out.read_text())["overall_flag"] is True


def test_groups_flags_go_together(tmp_path):
    X = _write(tmp_path, "x.npy", np.zeros((10, 2)))
    (tmp_path / "g.txt").write_text("\n".join("a b" .split() * 5))
    with pytest.raises(SystemExit, match="together"):
        main(["score", "--reference", X, "--deployment", X, "--reference-groups", str(tmp_path / "g.txt")])


@pytest.mark.parametrize("header,rows", [("auto", 4), ("yes", 3), ("no", 4)])
def test_numeric_header_choices(tmp_path, header, rows):
    path = tmp_path / "x.csv"
    path.write_text("1,2,3\n4,5,6\n7,8,9\n1,1,1\n")
    assert _load_features(str(path), header).shape == (rows, 3)
