"""
`WithinUnitHoldoutSchema` -- split único, determinístico, estratificado
por clase, dentro de una misma unidad. Simula orden temporal real de
llegada de trials (slicing secuencial, no aleatorio) -- semántica
"online simulado", distinta de WithinUnitKFoldSchema.
"""
from typing import List, Optional

from src.training.core.label_transform import ClassificationLabelTransform, LabelTransform
from src.training.data_provider import DataProvider
from src.training.splitting.fold import Fold
from src.training.splitting.index_split_strategy import IndexSplitStrategy, StratifiedSequentialSplit


class WithinUnitHoldoutSchema:
    def __init__(
        self, provider: DataProvider, ptrain: float, pval: float, ptest: float,
        units_to_test: Optional[List] = None,
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
        self.split_strategy = split_strategy or StratifiedSequentialSplit()
        self.label_transform = label_transform or ClassificationLabelTransform()

    def units_to_process(self) -> List:
        """El Schema decide qué unidades recorrer, con SU propio
        vocabulario (units_to_test), validado contra el provider."""
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
        """Nombres de partición que ESTA unidad va a generar, SIN
        generar folds completos (sin tocar el provider) -- para este
        Schema, siempre es una sola, predecible sin ningún dato real."""
        return [self.partition_name_for_unit(unit_id)]

    def partition_name_for_unit(self, unit_id) -> str:
        return f"unit_{unit_id:02d}"

    def generate_folds(self, **context) -> List[Fold]:
        """context esperado: unit_id."""
        unit_id = context["unit_id"]
        X, y = self.provider.get_unit(unit_id)
        n = len(y)

        # Paso 1: separar test. ntest como fracción del largo ORIGINAL de
        # y (no relativo al remanente de train).
        train_idx_intermediate, test_idx = self.split_strategy.split_indices(
            n_total=n, ntest=int(self.ptest * n), y=y
        )
        y_train_intermediate = y[train_idx_intermediate]

        # Paso 2: separar val del resto. Los índices se traducen de
        # vuelta al espacio del y ORIGINAL usando train_idx_intermediate
        # ANTES de perderlo -- mismo espacio de referencia que test_idx.
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
                "train_idx": [int(i) for i in train_idx],
                "val_idx": [int(i) for i in val_idx],
                "test_idx": [int(i) for i in test_idx],
                "dataloader_seed": self.dataloader_seed,
                **label_meta,
            },
        )]

    def partition_name(self, fold: Fold) -> str:
        return self.partition_name_for_unit(fold.metadata["unit_id"])