from .within_unit_holdout import WithinUnitHoldoutSchema
from .within_unit_kfold import WithinUnitKFoldSchema
from .leave_one_unit_out import LeaveOneUnitOutSchema

SCHEMA_REGISTRY = {
    "WithinUnitHoldoutSchema": WithinUnitHoldoutSchema,
    "WithinUnitKFoldSchema": WithinUnitKFoldSchema,
    "LeaveOneUnitOutSchema": LeaveOneUnitOutSchema,
}