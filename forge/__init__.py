"""Allora Forge — self-updating Topic 69 worker (24h BTC/USD).

A small MLOps package that retrains daily on fresh data, gates the new model
against the current one (auto-rollback on regression), versions every model,
and serves inferences to an Allora worker node.
"""
__version__ = "0.1.0"
