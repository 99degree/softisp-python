from .deshake_algo import (
    DeshakeAlgoBase,
    DeshakeAlgoV1
)

from .deshake_algo_v2 import (
    DeshakeAlgoV2
)

from .deshake_loop import (
    DeshakeLoopBase,
    DeshakeLoopV1
)

from .deshake_loop_v2 import (
    DeshakeLoopV2
)

from .deshake_applier import (
    DeshakeApplierBase,
    DeshakeApplierV1
)

from .deshake_applier_v2 import (
    DeshakeApplierV2
)

__all__ = [
    # Algorithms (3 classes)
    'DeshakeAlgoBase',
    'DeshakeAlgoV1',
    'DeshakeAlgoV2',
    
    # Control Loops / Coordinators (3 classes)
    'DeshakeLoopBase',
    'DeshakeLoopV1',
    'DeshakeLoopV2',
    
    # Appliers (3 classes)
    'DeshakeApplierBase',
    'DeshakeApplierV1',
    'DeshakeApplierV2',
]

# Total: 9 classes
# Architecture: Algo → Coordinator (with GDC fusion) → Applier
# V2: GoPro-grade with temporal smoothing, rolling shutter, mesh warp