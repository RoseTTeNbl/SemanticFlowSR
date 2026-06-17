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


def run_deap(X_train, y_train, X_test, y_test, **kw):
    """Classic tree-GP symbolic regression with DEAP (protected div, tournament + mut)."""
    import operator, math, random
    from deap import base, creator, gp, tools, algorithms

    n_vars = X_train.shape[1]
    pset = gp.PrimitiveSet("MAIN", n_vars)
    pset.addPrimitive(operator.add, 2); pset.addPrimitive(operator.sub, 2)
    pset.addPrimitive(operator.mul, 2)
    pset.addPrimitive(lambda a, b: a / b if abs(b) > 1e-6 else 1.0, 2, name="div")
    pset.addPrimitive(math.sin, 1); pset.addPrimitive(math.cos, 1)
    pset.addEphemeralConstant("rand", lambda: random.uniform(-1, 1))

    if not hasattr(creator, "FitnessMin"):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)
    tb = base.Toolbox()
    tb.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)
    tb.register("individual", tools.initIterate, creator.Individual, tb.expr)
    tb.register("population", tools.initRepeat, list, tb.individual)
    tb.register("compile", gp.compile, pset=pset)

    def evalfn(ind, X, y):
        f = tb.compile(expr=ind)
        try:
            pred = np.array([f(*row) for row in X], dtype=float)
        except (OverflowError, ValueError):
            return (1e9,)
        if not np.all(np.isfinite(pred)):
            return (1e9,)
        return (float(np.mean((pred - y) ** 2)),)

    tb.register("evaluate", evalfn, X=X_train, y=y_train)
    tb.register("select", tools.selTournament, tournsize=3)
    tb.register("mate", gp.cxOnePoint)
    tb.register("expr_mut", gp.genFull, min_=0, max_=2)
    tb.register("mutate", gp.mutUniform, expr=tb.expr_mut, pset=pset)
    tb.decorate("mate", gp.staticLimit(len, 17))
    tb.decorate("mutate", gp.staticLimit(len, 17))

    random.seed(kw.get("seed", 0))
    pop = tb.population(n=kw.get("population_size", 1000))
    hof = tools.HallOfFame(1)
    algorithms.eaSimple(pop, tb, cxpb=0.7, mutpb=0.2,
                        ngen=kw.get("generations", 40), halloffame=hof, verbose=False)
    best = tb.compile(expr=hof[0])
    pred = np.array([best(*row) for row in X_test], dtype=float)
    pred = np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
    return {"r2": r2_score(y_test, pred), "nmse": nmse(y_test, pred), "expression": str(hof[0])}


def run_dso(X_train, y_train, X_test, y_test, **kw):
    """DSR/DSO (deep symbolic regression, RL+RNN). Runs in the `dso37` env."""
    from dso import DeepSymbolicRegressor
    config = {"task": {"task_type": "regression",
                        "function_set": kw.get("function_set",
                            ["add", "sub", "mul", "div", "sin", "cos", "exp", "log", "const"])},
              "training": {"n_samples": kw.get("n_samples", 100000), "verbose": False}}
    model = DeepSymbolicRegressor(config)
    model.fit(X_train, y_train)
    pred = np.nan_to_num(np.asarray(model.predict(X_test), dtype=float))
    return {"r2": r2_score(y_test, pred), "nmse": nmse(y_test, pred),
            "expression": str(model.program_.pretty())}
