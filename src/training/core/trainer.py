"""
`Trainer_DL` -- ver docstring original para el resto del diseño (compone
CallbackDispatcher, loss_strategy/gradient_clipper opcionales, history
delegado al dispatcher).

Cambio de esta pasada: preload_to_device + device en el constructor.
create_dataset/create_dataloader (training/utils.py) NUNCA mueven nada a
GPU -- el dataset siempre vive en CPU. El movimiento a GPU es
responsabilidad EXCLUSIVA de Trainer_DL.model_infer, con dos modos:

  preload_to_device=False (default): el batch se mueve a `device` DENTRO
    de model_infer, en cada llamada -- estándar de la comunidad PyTorch,
    escala a datasets grandes (LeaveOneUnitOutSchema con pool de muchos
    sujetos) sin requerir que el split completo entre en memoria de GPU.

  preload_to_device=True: asume que quien arma el DataLoader YA movió los
    tensores a `device` de antemano (X_train_t = torch.tensor(...).to(device)
    antes de crear el TensorDataset) -- más rápido en tiempo de
    entrenamiento (sin transferencia CPU->GPU por batch), pero requiere
    que el split completo entre en memoria de GPU. Con este modo,
    model_infer NO vuelve a mover nada -- asume que X ya está en el
    device correcto.

`device` es obligatorio si preload_to_device=False (necesita saber a
dónde mover cada batch) -- si no se pasa y preload_to_device=False, se
asume que no hay que mover nada (equivalente a correr en CPU).
"""
from typing import Any, Callable, Dict, List, Optional

import torch

from src.training.core.callbacks.callback_dispatcher import CallbackDispatcher
from src.training.core.gradient_clippers import GradientClipper
from src.training.core.loss_strategy import LossStrategy


class Trainer_DL:
    def __init__(
        self,
        model,
        loss_fn,
        optimizer,
        train_loader,
        val_loader=None,
        max_epochs: int = 10,
        callbacks: Optional[List[Any]] = None,
        use_caching: bool = True,
        loss_strategy: Optional[LossStrategy] = None,
        gradient_clipper: Optional[GradientClipper] = None,
        preload_to_device: bool = False,
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            model: modelo de PyTorch a entrenar. Debe estar YA en el
                device correcto (model.to(device)) antes de pasarlo acá --
                Trainer_DL no mueve el modelo, solo los batches (y solo
                si preload_to_device=False).
            loss_fn: función de loss default.
            optimizer: optimizer.
            train_loader: DataLoader de entrenamiento. Sus tensores viven
                en CPU si preload_to_device=False (caso normal), o ya en
                `device` si preload_to_device=True (el caller los movió
                antes de armar el DataLoader).
            val_loader: DataLoader de validación (opcional), mismo
                criterio que train_loader.
            max_epochs: cantidad de épocas.
            callbacks: lista de instancias de callback YA RESUELTA por el
                orquestador.
            use_caching: preservado del original (no usado dentro de esta
                clase).
            loss_strategy: colaborador Strategy opcional. None (default)
                preserva `loss_fn(y_pred, y_true)` directo.
            gradient_clipper: colaborador opcional (NormClipper/ValueClipper).
                None (default) no aplica ningún clipping -- comportamiento
                preservado para configs existentes.
            preload_to_device: ver docstring del módulo. False (default) =
                batch a batch, escala mejor. True = el caller ya movió
                todo a `device` de antemano.
            device: a dónde mover cada batch cuando preload_to_device=False.
                None = no se mueve nada (equivalente a correr en CPU, o a
                que el caller ya haya movido los tensores por su cuenta
                sin usar preload_to_device=True explícitamente).
        """
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.best_model = None
        self.train_loader = train_loader

        self.val_loader = val_loader
        self.val_loader_flag = val_loader is not None

        self.max_epochs = max_epochs
        self.use_caching = use_caching
        self.loss_strategy = loss_strategy
        self.gradient_clipper = gradient_clipper
        self.preload_to_device = preload_to_device
        self.device = device

        self.stop_training = False

        self.dispatcher = CallbackDispatcher(callbacks)

    @property
    def history(self) -> Dict[str, Any]:
        """Delega a self.dispatcher.history -- ver callback_dispatcher.py."""
        return self.dispatcher.history

    def get_history(self) -> Dict[str, Any]:
        return self.dispatcher.history

    def get_model(self):
        return self.model

    def _compute_loss(self, y_pred, y_true, X):
        if self.loss_strategy is not None:
            return self.loss_strategy.compute(
                y_pred=y_pred, y_true=y_true, X=X, model=self.model,
                default_loss_fn=self.loss_fn,
            )
        return self.loss_fn(y_pred, y_true)

    def _maybe_move_target(self, y_true):
        """y_true no pasa por model_infer -- se mueve y castea acá aparte,
        con el mismo criterio de preload_to_device que el resto del batch.

        Cast a .long() si es un tensor entero-como-float (ej. float64/float32
        proveniente de un numpy sin dtype explícito) -- CrossEntropyLoss y la
        mayoría de losses de clasificación en PyTorch exigen y_true como
        LongTensor, no float. No se castea si y_true ya es float genuino
        (caso regresión, con IdentityLabelTransform -- ahí SÍ puede ser un
        target continuo real, forzar long lo rompería)."""
        if not self.preload_to_device and self.device is not None:
            y_true = y_true.to(self.device)

        if y_true.dtype in (torch.float32, torch.float64) and torch.equal(y_true, y_true.round()):
            y_true = y_true.long()

        return y_true

    def train(self) -> None:
        self.dispatcher.notify('on_train_begin', trainer=self)

        for epoch in range(self.max_epochs):
            self.dispatcher.notify('on_epoch_begin', trainer=self, epoch=epoch)
            self.model.train()
            for batch_idx, (X, y_true) in enumerate(self.train_loader):
                self.dispatcher.notify('on_batch_begin', trainer=self, batch_idx=batch_idx)

                y_true = self._maybe_move_target(y_true)

                self.optimizer.zero_grad()
                y_pred = self.model_infer(X)
                loss = self._compute_loss(y_pred, y_true, X)

                loss.backward()
                if self.gradient_clipper is not None:
                    self.gradient_clipper.clip(self.model.parameters())
                self.optimizer.step()

                self.dispatcher.notify(
                    'on_batch_end', trainer=self, batch_idx=batch_idx,
                    y_pred=y_pred, y_true=y_true, loss=loss.item(), training=True,
                )

            if self.val_loader_flag:
                self.evaluate(epoch)

            self.dispatcher.notify('on_epoch_end', trainer=self, epoch=epoch)

            if self.stop_training:
                break

        self.dispatcher.notify('on_train_end', trainer=self)

    def evaluate(self, epoch) -> None:
        self.model.eval()

        with torch.no_grad():
            for batch_idx, (X, y_true) in enumerate(self.val_loader):
                self.dispatcher.notify('on_batch_begin', trainer=self, batch_idx=batch_idx)

                y_true = self._maybe_move_target(y_true)

                y_pred = self.model_infer(X)
                loss = self._compute_loss(y_pred, y_true, X)

                self.dispatcher.notify(
                    'on_batch_end', trainer=self, batch_idx=batch_idx,
                    y_pred=y_pred, y_true=y_true, loss=loss.item(), training=False,
                )

    def infer(self, dataloader=None, X=None, probability: bool = True):
        self.model.eval()
        with torch.no_grad():
            if dataloader is not None:
                y_pred = []
                for batch_idx, (X_batch, y_true) in enumerate(dataloader):
                    y_pred.append(self.model_infer(X_batch))
                y_pred = torch.cat(y_pred, dim=0)
            else:
                if X is None:
                    raise ValueError("Either dataloader or X must be provided.")
                y_pred = self.model_infer(X)
        if not probability:
            y_pred = torch.argmax(y_pred, dim=1)
        return y_pred

    def model_infer(self, X):
        if not self.preload_to_device and self.device is not None:
            if isinstance(X, dict):
                X = {k: v.to(self.device) for k, v in X.items()}
            else:
                X = X.to(self.device)

        if isinstance(X, dict):
            X = {k: v.float() if torch.is_floating_point(v) else v for k, v in X.items()}
        elif torch.is_floating_point(X):
            X = X.float()

        if isinstance(X, dict):
            y_pred = self.model(**X)
        else:
            y_pred = self.model(X)
        return y_pred


class Trainer_ML:
    """Contrato duck-typed (sin heredar de Trainer_DL): `train()`,
    `infer(X, probability=True)`, `get_history()`.

    X_val/y_val opcionales, sin uso todavía dentro de la clase -- se
    dejan preparados en el constructor para no romper la interfaz cuando
    se implemente selección de mejor modelo/hiperparámetros vía val.
    Cualquier caller existente que instancie Trainer_ML(model, X_train,
    y_train) sin val sigue funcionando idéntico."""

    def __init__(self, model, X_train, y_train, X_val=None, y_val=None):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val

    def train(self) -> None:
        self.model.fit(self.X_train, self.y_train)

    def infer(self, X, probability: bool = True):
        if probability:
            return self.model.predict_proba(X)
        return self.model.predict(X)

    def get_history(self) -> Dict[str, Any]:
        return {}