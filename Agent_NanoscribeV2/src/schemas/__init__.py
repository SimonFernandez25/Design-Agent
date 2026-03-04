"""
Schemas Package - Named Object Geometry System

This package defines the core schemas for the named-object composition system:
- primitives: Native geometry types (box, cylinder) for executor boundary
- object_library: Named object definitions (geometry and composite types)
- assembly: Heterogeneous layout definitions
- edit_plan: Object-targeted redesign operations
"""

from schemas.primitives import (
    PRIMITIVE_SCHEMA,
    NATIVE_PRIMITIVE_TYPES,
    DERIVED_PRIMITIVE_TYPES,
    ALL_PRIMITIVE_TYPES,
    validate_primitive,
    is_native_primitive,
    is_derived_primitive
)

from schemas.object_library import (
    OBJECT_LIBRARY_SCHEMA,
    GEOMETRY_OBJECT_SCHEMA,
    COMPOSITE_OBJECT_SCHEMA,
    validate_object,
    validate_object_library,
    get_object_dependency_order
)

from schemas.assembly import (
    ASSEMBLY_SCHEMA,
    validate_assembly,
    get_grid_cell_positions
)

from schemas.edit_plan import (
    EDIT_PLAN_SCHEMA,
    EditScope,
    EditTargetType,
    parse_edit_target,
    validate_edit_plan,
    is_structural_edit
)

from schemas.validation import (
    v2_structural_gate,
    validate_v2_design,
    validate_assembly_references,
    validate_object_references,
    FORBIDDEN_FIELDS,
    REQUIRED_FIELDS
)

