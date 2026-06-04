"""Pytest configuration — mocks third-party modules that pull in heavyweight
optional dependencies not needed by the offline unit tests.

The package's airllm_base.py does ``from optimum.bettertransformer import
BetterTransformer`` at import time. Newer ``optimum`` releases moved that
module, so importing the ``airllm`` package can fail in a vanilla
environment even though none of the unit tests need it. We stub the symbol
out so the package imports cleanly and only the dispatch logic gets
exercised.
"""

import sys
from unittest.mock import MagicMock

# Stub the legacy optimum.bettertransformer module that airllm_base.py
# imports eagerly. The tests don't touch BetterTransformer.
if "optimum.bettertransformer" not in sys.modules:
    sys.modules["optimum.bettertransformer"] = MagicMock(
        BetterTransformer=MagicMock()
    )
