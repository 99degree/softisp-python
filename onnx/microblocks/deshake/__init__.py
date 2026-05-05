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
    DeshakeApplier
)

from .deshake_applier_v2 import (
    DeshakeApplierV2
)

from .deshake_mesh_grid import (
    DeshakeMeshGridBase
)

from .deshake_mesh_grid_simple import (
    DeshakeMeshGridSimple
)

from .deshake_mesh_grid_full import (
    DeshakeMeshGridFull
)

from .rotation_wrapper import (
    RotationState,
    RotationTrace
)

from .deshake_coordinator_cpu import (
    DeshakeCoordinatorCPU
)

__all__ = [
    # Stage 1: Algo Domain (3 classes)
    'DeshakeAlgoBase',
    'DeshakeAlgoV1',
    'DeshakeAlgoV2',
    
    # Stage 2: Coordinator Domain - Mesh Grid Generation (3 classes)
    'DeshakeMeshGridBase',
    'DeshakeMeshGridSimple',
    'DeshakeMeshGridFull',
    
    # Stage 3: Applier Domain - Grid Mapping (2 classes)
    'DeshakeApplier',
    'DeshakeApplierV2',
    
    # Legacy Control Loops (2 classes)
    'DeshakeLoopBase',
    'DeshakeLoopV1',
    'DeshakeLoopV2',
    
    # Universal Rotation Wrapper (2 classes)
    'RotationState',
    'RotationTrace',
    
    # Stateful CPU Coordinator (1 class)
    'DeshakeCoordinatorCPU',
]

# Total: 16 classes
# Industrial-Level API:
# Stage 1: Algo Domain → Output homography [3,3]
# Stage 2: Coordinator Domain → Output mesh_grid [mesh_h,mesh_w,2]
# Stage 3: Applier Domain → Apply grid mapping to frame