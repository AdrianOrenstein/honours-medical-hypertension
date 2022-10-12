from argparse import ArgumentParser
import time
from types import SimpleNamespace

from experiments.resnet import ClassificationExperiment
from loguru import logger
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

def get_cuda_config() -> Dict[str, Any]:
    if torch.cuda.is_available():
        return {
            "gpus": -1,
            "distributed_backend": "ddp",
        }

    else:
        logger.info("cuda not available")
        return {"distributed_backend": None}


def parse_args(experiment: pl.LightningModule):
    parser = ArgumentParser()
    parser.add_argument("--project_name", type=str, default="")
    parser.add_argument("--project_description", type=str, default="")
    parser.add_argument("--experiment_name", type=str, default="")
    parser.add_argument("--experiment_description", type=str, default=None)
    parser.add_argument("--pytorch_lightning_tune", type=bool, default=False)
    parser.add_argument("--balanced_dataset", type=str, default="False")

    logger.warning(f"Using ClassificationExperiment for now..")

    parser = experiment.add_model_specific_args(parser)

    parser = pl.Trainer.add_argparse_args(parser)

    parser.add_argument(
        "--SEED",
        type=int,
        default=int(time.time()),
    )

    parser.add_argument(
        "--test",
        type=bool,
        default=True,
    )

    return parser.parse_args()


if __name__ == "__main__":

    experiment_type = ClassificationExperiment

    args = parse_args(experiment=experiment_type)
    pl.seed_everything(seed=args.SEED)
    # logger.info(f"args.SEED = {args.SEED}")

    args = dict(vars(args))
    args.update(get_cuda_config())

    wandb_logger = WandbLogger(project="hypertension-project-unbalanced-test", log_model="all")

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        # filename="{epoch}-{train_loss:.2f}-{val_loss:.2f}",
        verbose=False,
        # save_last=True,
        # save_top_k=5,
    )

    args.update(
        {
            "logger": wandb_logger,
            "log_every_n_steps": 1,
            "callbacks": [checkpoint_callback],
        }
    )

    args = SimpleNamespace(**args)

    logger.info(f"{args=}")

    # log hyperparameters
    wandb_logger.experiment.config.update(dict(vars(args)))

    trainer = pl.Trainer.from_argparse_args(args, callbacks=args.callbacks)

    # logger.info(f"{args=}")

    experiment = experiment_type(**vars(args))

    trainer.fit(experiment, experiment.data_module)

    trainer.test(experiment, experiment.data_module)
