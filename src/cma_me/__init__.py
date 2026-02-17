"""
CMA-ME (MAP-Elites + CMA-ES) optimizer implementation.

This module provides a quality-diversity optimization algorithm that combines
MAP-Elites' archive-based diversity maintenance with CMA-ES's powerful local
search capabilities.
"""

from .optimizer import cma_me

__all__ = ['cma_me']
