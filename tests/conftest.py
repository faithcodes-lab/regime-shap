"""pytest configuration: select a non-interactive matplotlib backend.

pytest loads conftest.py before any test module, early enough to set the
backend before pyplot is imported, so the plot tests run headless (in CI too).
"""

import matplotlib

matplotlib.use("Agg")
