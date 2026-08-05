"""Gap analysis: manuscript shape parameter p = perimeter / sqrt(area).

Not computed anywhere in v1. IMPORTANT: `data_input.CELLS_INFO_COLUMNS`
already has a column literally named "P-Value", but that is a DIFFERENT
quantity (`4*pi / circularity**2`, see `data_input.extract_cell_info`) --
do not conflate the two. Signature + docstring only, no implementation this
round.
"""

import pandas as pd


def compute_shape_parameter_p(cells_info: pd.DataFrame) -> pd.Series:
    """Computes `p = Cell Perimeter / sqrt(Cell Area)`, the manuscript's
    Fig.5D/E/F shape parameter (lower = more isotropic/regular). Distinct
    from -- and must not be conflated with -- the existing `Cells_info.xlsx`
    `'P-Value'` column, which is `4*pi / circularity**2`, an unrelated
    formula already computed by `data_input.extract_cell_info`."""
    raise NotImplementedError
