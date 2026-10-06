"""
`WithinUnitKFoldSchema` -- K-fold aleatorio, estratificado por clase,
dentro de una misma unidad, para estimación robusta de performance vía
cross-validation. Algoritmo DISTINTO de WithinUnitHoldoutSchema (que es
secuencial y determinístico): la pregunta experimental es otra, y el
split es aleatorio. El test se separa UNA sola vez (aleatorio
estratificado, congelado) y es compartido por los K folds.
"""
from typing import List, Optional

import numpy as np
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

from src.training.core.label_transform import ClassificationLabelTransform, LabelTransform
from src.training.data_provider import DataProvider
from src.training.splitting.fold import Fold


class WithinUnitKFoldSchema:
    def __init__(
        self, provider: DataProvider, k_folds: int, ptest: float, seed: int,
        units_to_test: Optional[List] = None,
        label_transform: Optional[LabelTransform] = None,
    ):
        self.provider = provider
        self.k_folds = k_folds
        self.ptest = ptest
        self.seed = seed
        self.units_to_test = units_to_test
        self.label_transform = label_transform or ClassificationLabelTransform()

    def units_to_process(self) -> List:
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
        """K particiones predecibles SIN generar ningún fold real -- solo
        necesita self.k_folds, que ya conoce por constructor."""
        return [self.partition_name_for_fold(unit_id, fold_idx) for fold_idx in range(self.k_folds)]

    def partition_name_for_fold(self, unit_id, fold_idx: int) -> str:
        return f"unit_{unit_id:02d}_fold_{fold_idx}"

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
                    "train_idx": [int(i) for i in train_idx],
                    "val_idx": [int(i) for i in val_idx],
                    "test_idx": [int(i) for i in test_idx],
                    # Derivado de self.seed, distinto por fold -- para que
                    # el shuffling de DataLoaders no se comparta entre
                    # folds de la misma unidad (mismo motivo que el fix
                    # del Generator compartido entre unidades).
                    "dataloader_seed": self.seed * 1000 + fold_idx,
                    **label_meta,
                },
            ))
        return folds

    def partition_name(self, fold: Fold) -> str:
        return self.partition_name_for_fold(fold.metadata["unit_id"], fold.metadata["fold_idx"])