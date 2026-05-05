from .deshake_pre import (
    DeshakePre
)

from .deshake_core_simple import (
    DeshakeCoreSimple
)

from .deshake_core_full import (
    DeshakeCoreFull
)

from .deshake_post import (
    DeshakePost
)

# Legacy classes (kept for backward compatibility)
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
    # Refined 3-Stage Architecture (4 classes)
    'DeshakePre',           # Stage 1: Pre-processing (Algo Domain)
    'DeshakeCoreSimple',    # Stage 2: Core processing without IMU
    'DeshakeCoreFull',      # Stage 2: Core processing with IMU
    'DeshakePost',          # Stage 3: Post-processing (Applier Domain)
    
    # Legacy Classes (12 classes)
    'DeshakeAlgoBase',
    'DeshakeAlgoV1',
    'DeshakeAlgoV2',
    'DeshakeLoopBase',
    'DeshakeLoopV1',
    'DeshakeLoopV2',
    'DeshakeApplier',
    'DeshakeApplierV2',
    'DeshakeMeshGridBase',
    'DeshakeMeshGridSimple',
    'DeshakeMeshGridFull',
    'RotationState',
    'RotationTrace',
    'DeshakeCoordinatorCPU',
]

# Total: 16 classes
# Refined 3-Stage Architecture:
# Stage 1: DeshakePre → Calculate homography + GDC from VCM
# Stage 2: DeshakeCore → Generate mesh grid (Simple/Full)
# Stage 3: DeshakePost → Apply grid mapping to frame