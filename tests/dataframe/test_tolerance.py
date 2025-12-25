import importlib
import pytest


def test_tolerance_api_is_reexported():
    m = importlib.import_module("eb_evaluation.dataframe.tolerance")

    for name in [
        "TauEstimate",
        "TauMethod",
        "estimate_tau",
        "estimate_entity_tau",
        "hr_at_tau",
        "hr_auto_tau",
    ]:
        assert hasattr(m, name), f"Missing {name} from eb_evaluation.dataframe.tolerance"