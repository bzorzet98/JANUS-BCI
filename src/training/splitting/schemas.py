"""
`TrainingSchema` -- reemplaza el diseño anterior de Splitter + PartitionSpec
+ materialize_fold. Cada Schema unifica en una sola unidad lo que antes
eran dos responsabilidades separadas (decidir la partición + materializar
datos reales), recibiendo un `DataProvider` en su constructor en vez de
exponer un paso intermedio serializable hacia el caller.

Vocabulario: "unit" reemplaza a "subject" en TODO este módulo -- es el
único cambio de nombre que hace que este archivo sea reusable fuera de
BCI (ver src/training/data_provider.py para dónde vive el vocabulario
específico de EEG).

Cardinalidad uniforme entre los 3 Schemas: una llamada a generate_folds()
por "unidad de foco" (unit_id para los within-unit, test_unit para LOSO)
-- el caller (orquestador) itera, ningún Schema genera el barrido completo
de una sola llamada.

Contrato común (duck-typed, sin herencia real):
    class TrainingSchema(Protocol):
        def generate_folds(self, **context) -> List[Fold]: ...
        def partition_name(self, fold: Fold) -> str: ...

`label_transform` (LabelTransform, ver execution/label_transform.py) es un
colaborador opcional inyectado en cada Schema -- default
ClassificationLabelTransform() preserva el comportamiento actual
(codificación de clases) para configs existentes. Ningún Schema llama a
encode_labels directamente: eso mantendría el Schema atado a clasificación,
que es justo lo que este colaborador evita -- ver docstring de
label_transform.py.
"""
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

from src.training.data_provider import DataProvider
from src.training.core.label_transform import ClassificationLabelTransform, LabelTransform
from src.training.splitting.fold import Fold
from src.training.splitting.index_split_strategy import IndexSplitStrategy, StratifiedSequentialSplit

class WithinUnitHoldoutSchema:
    def __init__(
        self, provider: DataProvider, 
        ptrain: float, pval: float, ptest: float,
        units_to_test: Optional[List[Any]] = None,
        dataloader_seed: int = 8,
        split_strategy: Optional[IndexSplitStrategy] = None,
        label_transform: Optional[LabelTransform] = None,
    ):
        self.provider = provider
        self.ptrain = ptrain
        self.pval = pval
        self.ptest = ptest
        self.units_to_test = units_to_test
        self.dataloader_seed = dataloader_seed
        # Default explícito de ESTE Schema, no el default conceptual del
        # módulo -- ver docstring de index_split_strategy.py.
        self.split_strategy = split_strategy or StratifiedSequentialSplit()
        self.label_transform = label_transform or ClassificationLabelTransform()

    def units_to_process(self) -> List[Any]:
        """Reemplaza _resolve_units() del orquestador -- el Schema decide
        qué unidades recorrer, con SU propio vocabulario (units_to_test),
        validado contra el provider."""
        all_units = self.provider.list_units()
        if self.units_to_test is None:
            return all_units
        invalid = [u for u in self.units_to_test if u not in all_units]
        if invalid:
            raise ValueError(
                f"units_to_test pide unidades que el provider no tiene: {invalid}. "
                f"Unidades disponibles: {all_units}"
            )
        return self.units_to_test

    def generate_folds(self, **context) -> List[Fold]:
        unit_id = context["unit_id"]
        X, y = self.provider.get_unit(unit_id)
        n = len(y)

        train_idx_intermediate, test_idx = self.split_strategy.split_indices(n_total=n, ntest=int(self.ptest * n), y=y)
        y_train_intermediate = y[train_idx_intermediate]

        train_idx_relative, val_idx_relative = self.split_strategy.split_indices(
            n_total=len(train_idx_intermediate), ntest=int(self.pval * n), y=y_train_intermediate
        )
        train_idx = [train_idx_intermediate[i] for i in train_idx_relative]
        val_idx = [train_idx_intermediate[i] for i in val_idx_relative]

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        y_train_t, y_val_t, y_test_t, label_meta = self.label_transform.transform(y_train, y_val, y_test)

        return [Fold(
            X_train=X_train, y_train=y_train_t,
            X_val=X_val, y_val=y_val_t,
            X_test=X_test, y_test=y_test_t,
            metadata={
                "unit_id": unit_id,
                "dataloader_seed": self.dataloader_seed,
                "n_train": len(train_idx), "n_val": len(val_idx), "n_test": len(test_idx),
                "train_idx": [int(i) for i in train_idx],
                "val_idx": [int(i) for i in val_idx],
                "test_idx": [int(i) for i in test_idx],
                **label_meta,
            },
        )]

    def partition_name(self, fold: Fold) -> str:
        return f"unit_{fold.metadata['unit_id']:02d}"


class WithinUnitKFoldSchema:
    """K-fold aleatorio, estratificado por clase, dentro de una misma
    unidad -- para estimación robusta de performance vía cross-validation.
    Algoritmo DISTINTO de WithinUnitHoldoutSchema (que es secuencial y
    determinístico): acá la pregunta experimental es otra, y el split es
    aleatorio. NO reutiliza stratified_sequential_split. El test se separa
    UNA sola vez (aleatorio estratificado, congelado) y es compartido por
    los K folds -- no se re-separa test en cada fold."""

    def __init__(
        self, provider: DataProvider, k_folds: int, ptest: float, seed: int,
        units_to_test: Optional[List[Any]] = None,
        label_transform: Optional[LabelTransform] = None,
    ):
        self.provider = provider
        self.k_folds = k_folds
        self.ptest = ptest
        self.seed = seed
        self.units_to_test = units_to_test
        self.label_transform = label_transform or ClassificationLabelTransform()

    def units_to_process(self) -> List[Any]:
        """Reemplaza _resolve_units() del orquestador -- el Schema decide
        qué unidades recorrer, con SU propio vocabulario (units_to_test),
        validado contra el provider."""
        all_units = self.provider.list_units()
        if self.units_to_test is None:
            return all_units
        invalid = [u for u in self.units_to_test if u not in all_units]
        if invalid:
            raise ValueError(
                f"units_to_test pide unidades que el provider no tiene: {invalid}. "
                f"Unidades disponibles: {all_units}"
            )
        return self.units_to_test
    
    def generate_folds(self, **context) -> List[Fold]:
        """context esperado: unit_id."""
        unit_id = context["unit_id"]
        X, y = self.provider.get_unit(unit_id)

        all_idx = np.arange(len(y))
        test_splitter = StratifiedShuffleSplit(n_splits=1, test_size=self.ptest, random_state=self.seed)
        train_val_idx, test_idx = next(test_splitter.split(all_idx, y))

        y_train_val = y[train_val_idx]
        kfold = StratifiedKFold(n_splits=self.k_folds, shuffle=True, random_state=self.seed)

        X_test, y_test = X[test_idx], y[test_idx]

        folds = []
        for fold_idx, (train_relative, val_relative) in enumerate(kfold.split(train_val_idx, y_train_val)):
            train_idx = train_val_idx[train_relative]
            val_idx = train_val_idx[val_relative]

            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]

            y_train_t, y_val_t, y_test_t, label_meta = self.label_transform.transform(y_train, y_val, y_test)

            folds.append(Fold(
                X_train=X_train, y_train=y_train_t,
                X_val=X_val, y_val=y_val_t,
                X_test=X_test, y_test=y_test_t,
                metadata={
                    "unit_id": unit_id, "fold_idx": fold_idx, "k_folds": self.k_folds,
                    "dataloader_seed": self.seed * 1000 + fold_idx,
                    "n_train": len(train_idx), "n_val": len(val_idx), "n_test": len(test_idx),
                    "train_idx": [int(i) for i in train_idx],
                    "val_idx": [int(i) for i in val_idx],
                    "test_idx": [int(i) for i in test_idx],
                    **label_meta
                },
            ))
        return folds

    def partition_name(self, fold: Fold) -> str:
        """A diferencia de Holdout/LOSO (un Fold por llamada), acá varios
        Fold de la misma unidad comparten unit_id pero difieren en fold_idx
        -- el nombre de partición debe incluirlo para no pisar carpetas entre
        folds distintos de la misma unidad."""
        return f"unit_{fold.metadata['unit_id']:02d}_fold_{fold.metadata['fold_idx']}"


class LeaveOneUnitOutSchema:
    """Particiona a nivel de UNIDAD completa -- nunca mezcla trials de una
    misma unidad entre train y val. Una llamada a generate_folds = un
    fold (la unidad indicada en test_unit es el test; val_units es una
    muestra aleatoria de las unidades restantes). No toca directamente
    ningún dataset -- todo pasa por self.provider."""

    def __init__(
        self, provider: DataProvider, pval_units: float, seed: int,
        label_transform: Optional[LabelTransform] = None,
        units_to_test: List[int] = None, dataloader_seed: int = 8,
    ):
        self.provider = provider
        self.pval_units = pval_units
        self.seed = seed
        self.label_transform = label_transform or ClassificationLabelTransform()
        self.units_to_test = units_to_test
        self.dataloader_seed = dataloader_seed
        
    def units_to_process(self) -> List[Any]:
        """Reemplaza _resolve_units() del orquestador -- el Schema decide
        qué unidades recorrer, con SU propio vocabulario (units_to_test),
        validado contra el provider."""
        all_units = self.provider.list_units()
        if self.units_to_test is None:
            return all_units
        invalid = [u for u in self.units_to_test if u not in all_units]
        if invalid:
            raise ValueError(
                f"units_to_test pide unidades que el provider no tiene: {invalid}. "
                f"Unidades disponibles: {all_units}"
            )
        return self.units_to_test
    
    def generate_folds(self, **context) -> List[Fold]:
        """context esperado: unit_id (mismo nombre que los otros 2 Schemas,
        unificado para que el orquestador sea agnóstico -- semánticamente acá
        representa la unidad que se deja afuera como test, pero se pasa con
        la misma keyword que Holdout/KFold)."""
        unit_id = context["unit_id"]  # antes: context["test_unit"]
        remaining = [u for u in self.provider.list_units() if u != unit_id]

        rng = np.random.RandomState(self.seed)
        n_val = round(self.pval_units * len(remaining))
        # REGLA CRÍTICA, no negociable: ningún unit simultáneamente en
        # train_units y val_units -- la exclusión de abajo lo garantiza.
        if n_val > 0:
            val_units = [int(u) for u in rng.choice(remaining, size=n_val, replace=False)]
        else:
            val_units = []
        train_units = [u for u in remaining if u not in val_units]

        X_train, y_train = self.provider.get_units(train_units)
        if val_units:
            X_val, y_val = self.provider.get_units(val_units)
        else:
            X_val, y_val = None, None
        X_test, y_test = self.provider.get_unit(unit_id)

        y_train_t, y_val_t, y_test_t, label_meta = self.label_transform.transform(y_train, y_val, y_test)

        return [Fold(
            X_train=X_train, y_train=y_train_t,
            X_val=X_val, y_val=y_val_t,
            X_test=X_test, y_test=y_test_t,
            metadata={
                "n_train_units": len(train_units), "n_val_units": len(val_units), "n_test_units": 1,
                "n_train": len(y_train), "n_val": len(y_val) if y_val is not None else 0, "n_test": len(y_test),
                "test_unit": unit_id,
                "train_units": train_units,
                "val_units": val_units,
                "dataloader_seed": self.dataloader_seed,
                **label_meta,
            },
        )]

    def partition_name(self, fold: Fold) -> str:
        return f"loo_test-unit_{fold.metadata['test_unit']:02d}"


SCHEMA_REGISTRY = {
    "WithinUnitHoldoutSchema": WithinUnitHoldoutSchema,
    "WithinUnitKFoldSchema": WithinUnitKFoldSchema,
    "LeaveOneUnitOutSchema": LeaveOneUnitOutSchema,
}