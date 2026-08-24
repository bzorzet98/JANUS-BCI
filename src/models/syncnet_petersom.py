#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Define sincnet model.
@author: vpeterson
"""

import torch
import torch.nn as nn
from speechbrain.nnet.CNN import SincConv, Conv1d
from speechbrain.nnet.pooling import AdaptivePool
from speechbrain.nnet.normalization import LayerNorm, BatchNorm1d


# %%


class SincNetEncoder(torch.nn.Module):
    """
    Un encoder básico que utiliza SincConv como primera capa.

    """

    def __init__(self, sincconv_hparams, net_hparams):
        super().__init__()

        # 1. Capa SincConv (SincNet)
        self.sinc_conv = SincConv(
            **sincconv_hparams  # No se necesita padding especial aquí
        )

        # 2. Otros bloques CNN (siguiendo la arquitectura original de SincNet)
        self.relu = torch.nn.LeakyReLU(net_hparams["LeakyReluval"])
        batch_norm = net_hparams["batch_norm"]
        if batch_norm:
            self.norm1 = BatchNorm1d(input_size=net_hparams["in_channels"])
            self.norm2 = BatchNorm1d(input_size=net_hparams["cnn_channels"])
        else:
            self.norm1 = LayerNorm(net_hparams["in_channels"])
            self.norm2 = LayerNorm(net_hparams["cnn_channels"])

        # 3. Primer bloque Conv1d estándar
        self.cnn1d = Conv1d(
            in_channels=net_hparams["in_channels"],
            out_channels=net_hparams["cnn_channels"],
            kernel_size=net_hparams["cnn_kernelsize"],
            conv_init="kaiming",
        )

        self.cnnseq = nn.Sequential(
            *[
                Conv1d(
                    in_channels=net_hparams["cnn_channels"],
                    out_channels=net_hparams["cnn_channels"],
                    kernel_size=net_hparams["cnn_kernelsize"],
                    conv_init="kaiming",
                )
                for i in range(net_hparams["num_cnn_layers"])
            ]
        )

        # 4. Capa de Pooling (opcional)
        self.pool = AdaptivePool(output_size=net_hparams["pool_size"])

        self.classifier = nn.Sequential(
            nn.Linear(
                net_hparams["pool_size"] * net_hparams["cnn_channels"],
                net_hparams["fc_size"],
            ),
            nn.ReLU(),
            nn.Dropout(net_hparams["dropout"]),
            nn.Linear(net_hparams["fc_size"], net_hparams["n_class"]),
        )

    def forward(self, x):
        # 1. Aplicar SincConv
        # SpeechBrain espera [batch, time, channel]
        x = self.sinc_conv(x)  # [batch, channelsout]
        x = self.relu(x)
        x = self.norm1(x)
        # 2. Aplicar otras capas Conv1d
        x = self.cnn1d(x)
        x = self.cnnseq(x)
        x = self.relu(x)
        x = self.norm2(x)
        # print(x.shape)
        # 3. Aplicar pooling
        x = self.pool(x)
        # print(x.shape)
        # 4. Aplanar y pasar por capas fully connected
        x = x.reshape(x.size(0), -1)  # Aplanar
        # print(x.shape)
        x = self.classifier(x)

        return x


import speechbrain as sb


class SimpleBrain(sb.Brain):
    def __init__(
        self,
        modules=None,
        opt_class=None,
        hparams=None,
        run_opts=None,
        checkpointer=None,
        epoch_counter=None,
    ):
        super().__init__(
            modules=modules,
            opt_class=opt_class,
            hparams=hparams,
            run_opts=run_opts,
            checkpointer=checkpointer,
        )
        # Store the counter directly on the instance
        self.epoch_counter = epoch_counter
        loss_fn = lambda x: x
        self.val_loss_metric = sb.utils.metric_stats.MetricStats(
            metric={"loss": loss_fn}
        )
        self.train_loss_metric = sb.utils.metric_stats.MetricStats(
            metric={"loss": loss_fn}
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def compute_forward(self, batch, stage):
        data = batch[0].to(self.device).float()
        return self.modules.model(data)

    def compute_objectives(self, predictions, batch, stage):
        labels = batch[1].to(self.device).long()
        return torch.nn.functional.cross_entropy(predictions, labels)

    # def on_batch_end(self, batch, predictions, loss, stage):
    #     """Append loss to the correct metric tracker at the end of each batch."""
    #     if stage == sb.Stage.TRAIN:
    #         self.train_loss_metric.append(loss=loss.detach().item())
    #     elif stage == sb.Stage.VALID:
    #         self.val_loss_metric.append(loss=loss.detach().item())

    def on_stage_end(self, stage, stage_loss, epoch):
        """Called at the end of a training/validation/testing stage."""

        # We only save after the validation stage
        if stage == sb.Stage.TRAIN:
            self.train_loss_metric.clear()  # Reset train metric after each training epoch
        elif stage == sb.Stage.VALID:
            # 1. Retrieve the performance metric you care about (e.g., loss)
            current_performance = stage_loss
            print(f"Epoch {epoch}: Val Loss = {current_performance:.4f}")
            self.val_loss_metric.clear()

            # 2. Save the checkpoint
            # This will save the best model based on the 'min' direction
            # meta = {"loss": current_performance, "epoch": epoch}

            # try:
            #     self.checkpointer.save_and_keep_only(meta=meta, min_keys=["loss"])
            # except IndexError:
            #     # If saving fails on the first pass (due to empty internal lists), we skip this epoch's stopping logic
            #     print(
            #         f"Warning: Checkpoint saving skipped for Epoch {epoch} due to internal state issue. Early stopping is bypassed for this epoch."
            #     )
            #     return
            # best_loss_so_far = self.checkpointer.find_checkpoint(min_key="loss")[1][
            #     "loss"
            # ]
            # print(f"Best Val Loss so far: {best_loss_so_far:.4f}")

            # 3. Step the epoch counter using the correct attribute
            # limit_exceeded = current_performance >= best_loss_so_far

            self.epoch_counter.update_metric(current_performance)
            if self.epoch_counter.should_stop:
                print(
                    f"Early stopping triggered. No improvement in validation loss for {self.epoch_counter.limit_to_stop} consecutive epochs."
                )
                self.checkpointer.save_and_keep_only(
                    meta={"loss": current_performance, "epoch": epoch},
                    min_keys=["loss"],
                )
                # Optionally, you can also break out of training here if needed
                # self.hparams.stop_training = True
            # self.epoch_counter._save


def init_weights(m):
    """Applies Glorot/Xavier initialization to Conv and Linear layers."""
    if isinstance(m, SincConv):
        return  # Skip initialization for SincConv
    if isinstance(m, torch.nn.Linear):
        # Use Kaiming (He) initialization for ReLU, which is often better than Xavier for deep ReLU networks
        torch.nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")

        # Initialize bias to zero
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)

    # BatchNorm/LayerNorm are handled differently
    # elif isinstance(m, BatchNorm1d) or isinstance(m, LayerNorm):
    #     if m.weight is not None:
    #         torch.nn.init.constant_(m.weight, 1)
    #     if m.bias is not None:
    #         torch.nn.init.constant_(m.bias, 0)
