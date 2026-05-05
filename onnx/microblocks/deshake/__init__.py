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

from .deshake_applier_v3 import (
    DeshakeApplierV3
)

from .deshake_grid_generator import (
    DeshakeGridGenerator
)

from .rotation_wrapper import (
    RotationState,
    RotationTrace
)

from .deshake_coordinator_cpu import (
    DeshakeCoordinatorCPU
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
    
    # Appliers (4 classes)
    'DeshakeApplierBase',
    'DeshakeApplierV1',
    'DeshakeApplierV2',
    'DeshakeApplierV3',
    
    # Grid Generator (1 class)
    'DeshakeGridGenerator',
    
    # Universal Rotation Wrapper (2 classes)
    'RotationState',
    'RotationTrace',
    
    # Stateful CPU Coordinator (1 class)
    'DeshakeCoordinatorCPU',
]

# Total: 14 classes
# Architecture: Algo → Coordinator (CPU + ONNX) → Applier
# V3: Universal rotation wrapper + 16x16 grid generator