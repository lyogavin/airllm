"""Smoke-test: ``import airllm`` must succeed on a clean install.

Catches the class of regression described in issue #312 where a broken
``__init__.py`` made ``import airllm`` fail for fresh installs (3.0.x).
"""

import importlib


def test_import_airllm():
    """A plain ``import airllm`` must not raise."""
    mod = importlib.import_module("airllm")
    assert mod is not None


def test_base_model_class_exposed():
    from airllm import AirLLMBaseModel
    assert AirLLMBaseModel is not None


def test_auto_model_class_exposed():
    from airllm import AutoModel
    assert AutoModel is not None


def test_split_and_save_layers_exposed():
    from airllm import split_and_save_layers
    assert callable(split_and_save_layers)


def test_not_enough_space_exception_exposed():
    from airllm import NotEnoughSpaceException
    assert issubclass(NotEnoughSpaceException, Exception)
