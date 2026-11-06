"""
Rules that validate_config (src/utils/config_validation.py) applies to the
config of each stage before it runs (R14, STAGE_CONTRACTS.md section 0.4):
required fields, old-format keys with their replacement, and fields that
must match. Data only.

A required field is one the stage cannot run without. A field where null
has a meaning (for example subjects: null, all subjects) is never listed.
"""
from src.utils.config_validation import ConfigSpec

# Keys and modules of the format used before the per-stage scripts.
_OLD_FORMAT_KEYS = {
    "general_script_config": (
        "Its fields moved: model_init_seed to orchestrator, max_epochs to trainer.params, "
        "ptrain, pval, ptest and subjects_to_test (now units_to_test) to schema.params, "
        "and device to the --device flag. See projects/dummy-project/configs/training/ "
        "for the current format."
    ),
    "loaded_model_config": "Rename it to model_source.",
    "model_to_train": "Rename it to recipe_name.",
    "model_to_test": "Rename it to recipe_name.",
}
_OLD_MODULE_NAMES = {
    "src.networks": "Models are in src.models.",
    "src.torch_utils": "Torch datasets are in src.eeg_datasets and callbacks in src.training.core.callbacks.",
}

# project_config names the experiment folder and the dataset blocks decide
# what is loaded, so the folder name is the one to change.
_SAME_DATASET = (
    ("project_config.database_name", "databases.params.db_name"),
    ("project_config.database_session", "data_provider.params.default_session"),
)

_PROJECT_CONFIG = tuple(
    f"project_config.{key}"
    for key in ("project_name", "strategy_name", "recipe_name", "database_name", "database_session", "label")
)
# project_name and script_type of model_source have defaults in TestingOrchestrator.
_MODEL_SOURCE = tuple(
    f"model_source.{key}"
    for key in ("strategy_name", "recipe_name", "database_name", "database_session", "label")
)
_DATA_BLOCKS = (
    "databases.class_name", "databases.module_name", "databases.params",
    "data_provider.class_name", "data_provider.module_name",
)
_MODEL = ("model.class_name", "model.module_name", "model.params")
# Only the blocks: each one is either one entry per split or shared.
_TORCH_DATA = ("torch_dataset", "torch_dataloader")

_TRAIN_COMMON = _PROJECT_CONFIG + _DATA_BLOCKS + ("schema.class_name", "orchestrator.model_init_seed") + _MODEL
_TRAIN_DL_ONLY = (
    "optimizer.class_name", "optimizer.module_name", "optimizer.params",
    "criterion.class_name", "criterion.module_name", "criterion.params",
) + _TORCH_DATA

PREPROCESSING_SPECS = {
    "preprocessing": ConfigSpec(required=(
        "preprocessing_name",
        "database.class_name", "database.module_name", "database.kwargs",
        "preprocessing_pipeline.stages",
    )),
}

TRAINING_SPECS = {
    "train_dl": ConfigSpec(
        required=_TRAIN_COMMON + _TRAIN_DL_ONLY,
        old_keys=_OLD_FORMAT_KEYS, old_module_names=_OLD_MODULE_NAMES, same_value=_SAME_DATASET,
    ),
    "train_ml": ConfigSpec(
        required=_TRAIN_COMMON,
        old_keys=_OLD_FORMAT_KEYS, old_module_names=_OLD_MODULE_NAMES, same_value=_SAME_DATASET,
    ),
}

# test_dl and test_ml share the rules: the tester is chosen by
# model_source.script_type, not by the config's script_type.
_EVALUATION_SPEC = ConfigSpec(
    required=_PROJECT_CONFIG + _MODEL_SOURCE + _DATA_BLOCKS + _MODEL + _TORCH_DATA,
    old_keys=_OLD_FORMAT_KEYS, old_module_names=_OLD_MODULE_NAMES, same_value=_SAME_DATASET,
)
EVALUATION_SPECS = {"test_dl": _EVALUATION_SPEC, "test_ml": _EVALUATION_SPEC}
