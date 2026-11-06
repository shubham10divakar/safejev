"""Shared pieces (taxonomy, schema, fusion, calibration, registry).

Nothing here imports torch, so the core works with only ``pyyaml`` installed. These modules are
written to be liftable into a common ``jev-core`` package later (design doc §3).
"""
