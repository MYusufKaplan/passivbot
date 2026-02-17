"""
CMA-ES with Automatic Restarts optimizer implementation.

This module provides a CMA-ES optimizer that runs indefinitely with automatic
restarts from random locations when convergence is detected. It tracks the
global best solution across all restarts and supports checkpointing for
resume capability.
"""

from .optimizer import cma_es_restarts

__all__ = ['cma_es_restarts']
