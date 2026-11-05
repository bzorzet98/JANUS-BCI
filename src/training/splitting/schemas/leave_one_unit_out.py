"""
`LeaveOneUnitOutSchema` -- particiona a nivel de UNIDAD completa -- nunca
mezcla trials de una misma unidad entre train y val. Una llamada a
generate_folds = un fold (la unidad indicada en unit_id es el test;
val_units es una muestra aleatoria de las unidades restantes). No toca
directamente ningún dataset -- todo pasa por self.provider.
"""
from typing import List, Optional

import numpy as np

from src.training.core.label_transform import ClassificationLabelTransform, LabelTransform
from src.training.data_provider import DataProvider
from src.training.splitting.fold import Fold
from src.training.splitting.partition_naming import build_partition_name


class LeaveOneUnitOutSchema:
    def __init__(
        self, provider: DataProvider, pval_units: float, seed: int,
        units_to_test: Optional[List] = None,
        dataloader_seed: int = 8,
        label_transform: Optional[LabelTransform] = None,
    ):
        self.provider = provider
        self.pval_units = pval_units
        self.seed = seed
        self.units_to_test = units_to_test
        self.dataloader_seed = dataloader_seed
        self.label_transform = label_transform or ClassificationLabelTransform()

    def units_to_process(self) -> List:
        """Acá 'unidades a procesar' son los candidatos a ser dejados
        afuera como test -- mismo vocabulario/validación que los otros
        2 Schemas, pero semánticamente distinto (cada uno genera SU
        propio fold con el resto como train/val)."""
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

    def expected_partition_names(self, unit_id) -> List[str]:
        """Una sola partición por unidad de test, predecible sin tocar
        el provider."""
        return [self.partition_name_for_unit(unit_id)]

    def partition_name_for_unit(self, unit_id) -> str:
        return build_partition_name(unit_id, 0)

    def generate_folds(self, **context) -> List[Fold]:
        """context esperado: unit_id (UN identificador, no una lista --
        misma cardinalidad que los otros 2 Schemas: una llamada por
        unidad de foco)."""
        unit_id = context["unit_id"]
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
                "test_unit": unit_id,
                "train_units": train_units,
                "val_units": val_units,
                "dataloader_seed": self.dataloader_seed,
                **label_meta,
            },
        )]

    def partition_name(self, fold: Fold) -> str:
        return self.partition_name_for_unit(fold.metadata["test_unit"])