import pandas as pd
import pytest

from src.analyze import main


def _write_csv(tmp_path, df: pd.DataFrame, name: str = "data.csv"):
    path = tmp_path / name
    df.to_csv(path, index=False)
    return path


def test_cli_prints_full_report(tmp_path, monkeypatch, capsys):
    df = pd.DataFrame(
        {
            "country": ["AR", "CL"] * 10,
            "revenue": [float(i) for i in range(20)],
        }
    )
    path = _write_csv(tmp_path, df)

    monkeypatch.setattr("sys.argv", ["analyze", str(path)])
    main()

    output = capsys.readouterr().out

    assert "EXPLAIN MY DATA" in output
    assert "PROFILING" in output
    assert "QUALITY" in output
    assert "STATISTICS" in output
    assert "RELATIONSHIPS" in output
    assert "ANALYSIS CANDIDATES" in output
    assert "Analysis complete." in output


def test_cli_rejects_missing_file(tmp_path, monkeypatch):
    missing = tmp_path / "nope.csv"

    monkeypatch.setattr("sys.argv", ["analyze", str(missing)])

    with pytest.raises(SystemExit, match="file not found"):
        main()


def test_cli_rejects_non_csv_file(tmp_path, monkeypatch):
    path = tmp_path / "data.xlsx"
    path.write_bytes(b"not a csv")

    monkeypatch.setattr("sys.argv", ["analyze", str(path)])

    with pytest.raises(SystemExit, match="must be a CSV"):
        main()


def test_cli_rejects_unreadable_csv(tmp_path, monkeypatch):
    path = tmp_path / "data.csv"
    path.write_bytes(b"\xff\xfe\xfdinvalid utf-8")

    monkeypatch.setattr("sys.argv", ["analyze", str(path)])

    with pytest.raises(SystemExit, match="Error reading CSV"):
        main()


def test_cli_rejects_empty_dataset(tmp_path, monkeypatch):
    path = tmp_path / "empty.csv"
    path.write_text("")

    monkeypatch.setattr("sys.argv", ["analyze", str(path)])

    with pytest.raises(SystemExit, match="empty"):
        main()
