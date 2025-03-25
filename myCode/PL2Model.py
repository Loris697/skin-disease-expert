import torch
import torch.nn as nn
import pytorch_lightning as pl
# For Lightning 2.0 or above
# import lightning.pytorch as pl

from sklearn.metrics import balanced_accuracy_score

try:
    import madgrad
except ImportError:
    print("madgrad not installed. Please install if needed.")

try:
    from radam import RAdam
except ImportError:
    print("RAdam not installed. Please install if needed.")

class PLModel(pl.LightningModule):
    def __init__(
        self,
        name,
        model,
        loss=None,
        num_epochs=0,
        path=None,
        lr=0.001,
        wd=0.0,
        optimName='MADGRAD',
        example_input_array = torch.randn(1, 3, 256, 256)
    ):
        super().__init__()

        self.name = name
        self.model = model

        # Hyperparameters and others
        self.lr = lr
        self.wd = wd
        self.optimName = optimName
        self.epoch = num_epochs
        self.num_classes = 8

        # Loss
        if loss is None:
            print("No loss specified, using default CrossEntropyLoss.")
            self.loss = nn.CrossEntropyLoss()
        else:
            self.loss = loss

        # Keep lists to accumulate validation preds/targets over entire epoch
        self.val_preds = []
        self.val_targets = []

        self.train_preds = []
        self.train_targets = []

        # To log the graph
        self.example_input_array = example_input_array

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        if self.optimName == 'SGD':
            optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
        elif self.optimName == 'MADGRAD':
            optimizer = madgrad.MADGRAD(
                self.parameters(),
                lr=self.lr,
                momentum=0.9,
                weight_decay=self.wd,
                eps=1e-06
            )
        else:
            optimizer = RAdam(
                self.parameters(),
                lr=self.lr,
                weight_decay=self.wd
            )

        return {
            "optimizer": optimizer
        }

    # ------------------
    # TRAINING
    # ------------------
    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        loss = self.loss(out, y)

        ## Convert logits to predicted class indices
        preds = torch.argmax(out, dim=1)

        # Store for balanced accuracy calculation at epoch end
        self.train_preds.append(preds.cpu())
        self.train_targets.append(y.cpu())

        # Log step-wise (or epoch-wise) metrics
        # Here, we log the loss on every step so we can see it in the progress bar
        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True)
        
        # Return the loss for backprop
        return loss

    def on_train_epoch_end(self):
        # Concatenate all predictions and targets
        all_preds = torch.cat(self.train_preds).numpy()
        all_targets = torch.cat(self.train_targets).numpy()

        # Compute balanced accuracy using sklearn
        val_bal_acc = balanced_accuracy_score(all_targets, all_preds)

        # Log the balanced accuracy
        self.log("train_val_acc", val_bal_acc, on_epoch=True, prog_bar=True)

        # Clear the lists for the next epoch
        self.train_preds.clear()
        self.train_targets.clear()

    # ------------------
    # VALIDATION
    # ------------------
    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        loss = self.loss(out, y)

        # Convert logits to predicted class indices
        preds = torch.argmax(out, dim=1)

        # Store for balanced accuracy calculation at epoch end
        self.val_preds.append(preds.cpu())
        self.val_targets.append(y.cpu())

        # Log validation loss on the step
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        return loss

    def on_validation_epoch_end(self):
        # Concatenate all predictions and targets
        all_preds = torch.cat(self.val_preds).numpy()
        all_targets = torch.cat(self.val_targets).numpy()

        # Compute balanced accuracy using sklearn
        val_bal_acc = balanced_accuracy_score(all_targets, all_preds)

        # Log the balanced accuracy
        self.log("bal_val_acc", val_bal_acc, on_epoch=True, prog_bar=True)

        # Clear the lists for the next epoch
        self.val_preds.clear()
        self.val_targets.clear()

    # ------------------
    # HOOK FOR LOGGING GRADIENTS & PARAMETER NORMS
    # ------------------
    def on_after_backward(self):
        """
        Called after loss.backward() in the training loop.
        Here we log:
          - Gradients histograms
          - Parameter norms
          - Gradient norms
        """
        # Skip if there's no logger or if we are not training
        if not self.logger:
            return

        # current global step
        global_step = self.global_step

        for name, param in self.named_parameters():
            # Only log for parameters that require grad
            if param.requires_grad and param.grad is not None:
                # Log gradient histogram
                self.logger.experiment.add_histogram(
                    f"gradients/{name}",
                    param.grad,
                    global_step
                )

                # Log parameter norm (L2)
                param_norm = param.detach().data.norm(2)
                self.logger.experiment.add_scalar(
                    f"param_norms/{name}",
                    param_norm,
                    global_step
                )

                # Log gradient norm (L2)
                grad_norm = param.grad.detach().data.norm(2)
                self.logger.experiment.add_scalar(
                    f"grad_norms/{name}",
                    grad_norm,
                    global_step
                ) 

    