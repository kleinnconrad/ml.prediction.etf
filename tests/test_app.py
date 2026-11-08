"""
Offline test of the Streamlit app (src/app.py): the page starts without a run, and a run on a
synthetic price panel (instead of the download) shows the gate, the metrics, the ranking and
the details of one ETF.
"""
import sys
from pathlib import Path
from unittest import mock

import pytest
from streamlit.testing.v1 import AppTest

from conftest import ohlc_panel

APP = Path(__file__).resolve().parents[1] / 'src' / 'app.py'


def test_the_app_shows_the_ranking_after_a_run(monkeypatch: pytest.MonkeyPatch) -> None:
    # AppTest leaves the app script registered as __main__; worker processes started later with
    # the 'spawn' method (the variable review) would re-import it. The original is restored.
    monkeypatch.setitem(sys.modules, '__main__', sys.modules['__main__'])
    panel = ohlc_panel(n_etfs=40, n_days=1800, seed=5)
    with mock.patch('prices.fetch_price_panel', return_value=panel) as download:
        app = AppTest.from_file(str(APP), default_timeout=120)
        app.run()
        assert not app.exception
        assert any('Start the analysis' in info.value for info in app.info)
        download.assert_not_called()

        app.button[0].click().run()
        assert not app.exception
        download.assert_called_once()

    assert len(app.success) + len(app.error) == 1          # the gate message
    labels = [metric.label for metric in app.metric]
    assert labels[:4] == ['Ranked ETFs', 'Up', 'Validation AUC', 'Top-share hit rate']
    assert app.metric[0].value == '40'
    assert [tab.label for tab in app.tabs] == ['Ranking', 'ETF Details', 'Validation', 'Model']
    ranking = app.dataframe[0].value
    assert len(ranking) == 40 and list(ranking['rank'][:3]) == [1, 2, 3]
    assert app.selectbox[0].value in set(panel.close.columns)
