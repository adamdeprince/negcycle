"""Top-level Python import surface for arbcycle.

Keep this file small. The idea is that Python users import `arbcycle`, and you
can later add conditional imports / namespace plumbing here without moving the C++
extension layout around.
"""

from arbcycle._common import Cycle, Edge
from arbcycle._arbcycle_native import detect_best_backend
# from arbcycle._generic import Cycle, Edge, ArbitrageDetector # from arbcycle.

# from ._arbcycle_native import BackendKind, available_backends, detect_best_backend

__all__ = [
   "detect_best_backend",
   "Cycle",
   "Edge",
   "ArbitrageDetector"
]
