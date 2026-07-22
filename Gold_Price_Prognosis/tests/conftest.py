"""Tests never need an on-screen window; force a non-interactive matplotlib
backend before any test imports pyplot, avoiding this environment's flaky
TkAgg/tk.tcl availability once enough figures have been created across the suite.
"""
import matplotlib
matplotlib.use("Agg")
