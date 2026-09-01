from typing import Literal

import numpy as np

type Vector3D = np.ndarray[tuple[Literal[3]], np.dtype[np.float64]]
type State = np.ndarray[tuple[Literal[7]], np.dtype[np.float64]]
