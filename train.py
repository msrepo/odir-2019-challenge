import torch
from torch import nn
from torch.utils.data import DataLoader
from datasets.custom_dataset import CSVDataset

from ignite.engine import Engine, Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss
from ignite.handlers import ModelCheckpoint
from ignite.contrib.handlers import TensorboardLogger, global_step_from_engine

from transforms.transforms import get_img_transform,label_transform
from models.models import Net


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 300
TRAIN_DATA_ROOT_DIR = 'odir2019/ODIR-5K_Training_Dataset'
TRAIN_CSV= 'csv/processed_train_ODIR-5K.csv'
VAL_CSV = 'csv/processed_val_ODIR-5K.csv'
N_CLASSES = 8
N_CHANNELS = 3
LOG_INTERVAL = 20
BATCH_SIZE = 32
LEARNING_RATE = 5e-2
MAX_EPOCHS = 100
N_CHECKPOINTS = 5

model = Net().to(device)

data_transform = get_img_transform(img_size = IMG_SIZE)

train_loader = DataLoader(
    CSVDataset(data_root_dir=TRAIN_DATA_ROOT_DIR,csv_path=TRAIN_CSV,img_transform=data_transform,
               label_transform=label_transform), batch_size=BATCH_SIZE, shuffle=True
)

val_loader = DataLoader(
    CSVDataset(data_root_dir=TRAIN_DATA_ROOT_DIR,csv_path=VAL_CSV,img_transform=data_transform,
               label_transform=label_transform), batch_size=BATCH_SIZE, shuffle=False
)


optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

trainer = create_supervised_trainer(model, optimizer, criterion, device)

val_metrics = {
    "accuracy": Accuracy(),
    "loss": Loss(criterion)
}

train_evaluator = create_supervised_evaluator(model, metrics=val_metrics, device=device)
val_evaluator = create_supervised_evaluator(model, metrics=val_metrics, device=device)

log_interval = LOG_INTERVAL

@trainer.on(Events.ITERATION_COMPLETED(every=log_interval))
def log_training_loss(engine):
    print(f"Epoch[{engine.state.epoch}], Iter[{engine.state.iteration}] Loss: {engine.state.output:.2f}")

@trainer.on(Events.EPOCH_COMPLETED)
def log_training_results(trainer):
    train_evaluator.run(train_loader)
    metrics = train_evaluator.state.metrics
    print(f"Training Results - Epoch[{trainer.state.epoch}] Avg accuracy: {metrics['accuracy']:.2f} Avg loss: {metrics['loss']:.2f}")


@trainer.on(Events.EPOCH_COMPLETED)
def log_validation_results(trainer):
    val_evaluator.run(val_loader)
    metrics = val_evaluator.state.metrics
    print(f"Validation Results - Epoch[{trainer.state.epoch}] Avg accuracy: {metrics['accuracy']:.2f} Avg loss: {metrics['loss']:.2f}")


def score_function(engine):
    return engine.state.metrics["accuracy"]


model_checkpoint = ModelCheckpoint(
    "checkpoint",
    n_saved=N_CHECKPOINTS,
    filename_prefix="best",
    score_function=score_function,
    score_name="accuracy",
    global_step_transform=global_step_from_engine(trainer),
    require_empty=False,
)
  
val_evaluator.add_event_handler(Events.COMPLETED, model_checkpoint, {"model": model})

tb_logger = TensorboardLogger(log_dir="tb-logger")

tb_logger.attach_output_handler(
    trainer,
    event_name=Events.ITERATION_COMPLETED(every=100),
    tag="training",
    output_transform=lambda loss: {"batch_loss": loss},
)

for tag, evaluator in [("training", train_evaluator), ("validation", val_evaluator)]:
    tb_logger.attach_output_handler(
        evaluator,
        event_name=Events.EPOCH_COMPLETED,
        tag=tag,
        metric_names="all",
        global_step_transform=global_step_from_engine(trainer),
    )

trainer.run(train_loader, max_epochs=100)

tb_logger.close()
