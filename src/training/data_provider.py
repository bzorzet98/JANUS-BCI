"""
`DataProvider` -- contrato genérico, agnóstico de dominio, para obtener
datos de una o varias "unidades" (lo que sea que eso signifique en cada
dominio: sujeto en BCI, paciente, empresa, etc.). Ningún `TrainingSchema`
sabe qué es una "unidad" en términos concretos -- solo sabe pedirle datos
a un `DataProvider` por identificador.

`BCIDataProviderAdapter` es EL ÚNICO lugar de todo `src/training/` donde
aparece vocabulario específico de EEG/BCI ("subject", "session") -- envuelve
un dataset BCI (PreprocessedDataset u otro con la misma interfaz de
flatten_subject_data/flatten_pool_data) para exponer el contrato genérico
que el resto del framework espera. Si mañana se usa este framework para
otro dominio, este es el único archivo que se reemplaza por un adaptador
análogo -- nada de execution/ ni splitting/ necesita cambiar.
"""
from typing import Any, List, Protocol, Tuple


class DataProvider(Protocol):
    """Contrato mínimo que cualquier fuente de datos debe cumplir para
    poder usarse con los TrainingSchema de src/training/."""

    def get_unit(self, unit_id: Any, **kwargs) -> Tuple[Any, Any]:
        """Devuelve (X, y) para UNA unidad."""
        ...

    def get_units(self, unit_ids: List[Any], **kwargs) -> Tuple[Any, Any]:
        """Devuelve (X, y) agregados para VARIAS unidades."""
        ...

    def list_units(self) -> List[Any]:
        """Devuelve la lista de identificadores de unidad disponibles."""
        ...