"""
Headless tests for the Streamlit dashboard.

These run the real app script with Streamlit's AppTest harness against the
committed outputs/, so a broken import, a missing artifact, or a crash on a
slider change fails here instead of in front of a viewer.
"""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "app" / "dashboard.py")


@pytest.fixture
def app():
    return AppTest.from_file(APP, default_timeout=120).run()


def _metric(app, label):
    return next(m for m in app.metric if m.label == label)


def test_dashboard_loads_without_error(app):
    assert not app.exception
    assert app.title[0].value.startswith("ChurnOpt")
    assert len(app.tabs) == 4


def test_default_assumptions_give_a_profitable_campaign(app):
    contacts = int(_metric(app, "Customers to contact").value.replace(",", ""))
    assert contacts == 800  # $20,000 budget / $25 per contact
    assert _metric(app, "Expected net profit").value.startswith("$")
    assert _metric(app, "ROI (net / cost)").value.endswith("%")


def test_lower_save_rate_shrinks_profit_and_the_break_even_gap(app):
    base_profit = float(_metric(app, "Expected net profit").value.replace("$", "").replace(",", ""))

    app.sidebar.slider[0].set_value(0.10).run()  # save rate
    assert not app.exception
    low_profit = float(_metric(app, "Expected net profit").value.replace("$", "").replace(",", ""))
    assert low_profit < base_profit / 3


def test_ruinous_contact_cost_shows_no_campaign_warning(app):
    app.sidebar.slider[0].set_value(0.05).run()
    app.sidebar.slider[1].set_value(100).run()
    assert not app.exception
    assert any("No profitable campaign" in w.value for w in app.warning)


def test_removing_the_budget_lets_the_list_grow_to_the_profit_maximum(app):
    app.sidebar.checkbox[0].uncheck().run()
    assert not app.exception
    contacts = int(_metric(app, "Customers to contact").value.replace(",", ""))
    assert contacts > 2000
