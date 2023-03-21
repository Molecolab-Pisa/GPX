import pytest  # noqa
import jax
import gpx  # noqa


def test_float64():
    """check that importing gpx changes jax default to double precision"""
    assert jax.config.x64_enabled is True
