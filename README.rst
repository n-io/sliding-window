Synchronous Slide Tests
=======================

This repo contains several sample programs building up primitives
for a synchronous slide FFT.

This code is compatible with Cerebras SDK 1.0.0.

synchronous-slide-right
-----------------------
Constructs a program with a row of ``kernel_width`` PEs, where ``kernel_width``
is a multiple of four from 8 to 32.
Each PE has arrays ``arr0`` and ``arr1`` of size ``num_elems``.
Each PE from 0 to ``kernel_width`` / 2 - 1 initializes ``arr0``.
The kernel shifts the data by ``kernel_width`` / 4 PEs, with the shifted
data ending up in ``arr1``.
Thus, for PE i, where i < ``kernel_width`` / 2, the contents of ``arr0``
are moved to ``arr1`` of PE i + ``kernel_width`` / 4.

synchronous-slide
-----------------
Same as above, except with data shifting in both directions.
For PE i, where i < ``kernel_width`` / 2, the contents of ``arr0``
are moved to ``arr1`` of PE i + ``kernel_width`` / 4.
For PE i, where i >= ``kernel_width`` / 2, the contents of ``arr0``
are moved to ``arr2`` of PE i - ``kernel_width`` / 4.

The ``sweep.py`` script performs a sweep over multiple kernel widths and
number of elements, producing a CSV file with cycle counts on the left,
middle, and right PEs for performing the synchronous slide.
