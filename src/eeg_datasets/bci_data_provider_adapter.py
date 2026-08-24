class BCIDataProviderAdapter:
    """Envuelve un dataset BCI (PreprocessedDataset u otro con la misma
    interfaz) para exponer el contrato DataProvider genérico. Es EL ÚNICO
    lugar donde 'session'/'subject' aparecen fuera de eeg_datasets/."""

    def __init__(self, dataset, default_session: str):
        self.dataset = dataset
        self.default_session = default_session

    def get_unit(self, unit_id, **kwargs):
        session = kwargs.get("session", self.default_session)
        X, y , metadata = self.dataset.flatten_subject_data(unit_id, session=session)
        return X, y

    def get_units(self, unit_ids, **kwargs):
        session = kwargs.get("session", self.default_session)
        X, y , metadata = self.dataset.flatten_pool_data(unit_ids, session=session)
        return X, y

    def list_units(self):
        return self.dataset.subject_list