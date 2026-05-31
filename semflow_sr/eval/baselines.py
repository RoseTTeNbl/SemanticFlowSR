"""Baseline regressor adapters (sklearn-style fit/predict) used by run_*_baseline scripts.

These import their heavy deps lazily so the core package does not require them. Each
baseline is expected to run in its OWN conda env (see docs/BASELINES.md).
"""
from __future__ import annotations
import numpy as np

from .metrics import r2_score, nmse


def run_pysr(X_train, y_train, X_test, y_test, **kw):
    from pysr import PySRRegressor
    model = PySRRegressor(
        niterations=kw.get("niterations", 100),
        binary_operators=kw.get("binary_operators", ["+", "-", "*", "/"]),
        unary_operators=kw.get("unary_operators", ["sin", "cos", "exp", "sqrt"]),
        progress=False, verbosity=0,
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    return {"r2": r2_score(y_test, pred), "nmse": nmse(y_test, pred),
            "expression": str(model.sympy())}


def run_gplearn(X_train, y_train, X_test, y_test, **kw):
    from gplearn.genetic import SymbolicRegressor
    model = SymbolicRegressor(
        population_size=kw.get("population_size", 1000),
        generations=kw.get("generations", 20),
        function_set=kw.get("function_set", ("add", "sub", "mul", "div", "sin", "cos")),
        verbose=0,
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    return {"r2": r2_score(y_test, pred), "nmse": nmse(y_test, pred),
            "expression": str(model._program)}
