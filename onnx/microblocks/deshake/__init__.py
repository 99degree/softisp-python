from .deshake_pre import (
    DeshakePre
)

from .deshake_core_simple import (
    DeshakeCoreSimple,
    DeshakeCoreSimple16,
    DeshakeCoreSimple32
)

from .deshake_core_full import (
    DeshakeCoreFull,
    DeshakeCoreFull16,
    DeshakeCoreFull32
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
    # Refined 3-Stage Architecture (8 classes)
    
    # Stage 1: Pre-processing (Algo Domain)
    'DeshakePre',           # Calculate homography + GDC from VCM
    
    # Stage 2: Core processing (Coordinator Domain)
    'DeshakeCoreSimple',    # Core processing without IMU (adaptive mesh size)
    'DeshakeCoreSimple16',  # Core processing without IMU (16x16 mesh)
    'DeshakeCoreSimple32',  # Core processing without IMU (32x32 mesh)
    'DeshakeCoreFull',      # Core processing with IMU (adaptive mesh size)
    'DeshakeCoreFull16',    # Core processing with IMU (16x16 mesh)
    'DeshakeCoreFull32',    # Core processing with IMU (32x32 mesh)
    
    # Stage 3: Post-processing (Applier Domain)
    'DeshakePost',          # Apply grid mapping to frame
    
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

# Total: 20 classes
# Refined 3-Stage Architecture:
# Stage 1: DeshakePre → Calculate homography + GDC from VCM
# Stage 2: DeshakeCore → Generate mesh grid (Simple/Full, 16x16/32x32)
# Stage 3: DeshakePost → Apply grid mapping to frame
#
# Mesh Size Options:
# - DeshakeCoreSimple: Adaptive (default 16x16)
# - DeshakeCoreSimple16: Fixed 16x16 (compile-time constant)
# - DeshakeCoreSimple32: Fixed 32x32 (compile-time constant)
# - DeshakeCoreFull: Adaptive (default 16x16)
# - DeshakeCoreFull16: Fixed 16x16 (compile-time constant)
# - DeshakeCoreFull32: Fixed 32x32 (compile-time constant)