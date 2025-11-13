import time
import logging
import argparse
from config import get_config, PicoGPTConfig, TrainConfig, from_dict
from dataset import DataLoaderLite
from model import load_model
import torch
import torch.distributed as dist
import math
from utils import create_timer
# from preset import GPT2Preset, GPT2XLConfig

def get_logger():
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(__name__)


log = get_logger()


def append_file_handler(logger, config: PicoGPTConfig):
    if config.save_log:
        filename = f"{config.logging_dir_prefix}/log-{config.execution_id}.log"
        file_handler = logging.FileHandler(filename, "w")
        file_handler.setLevel(logging.INFO)
        logger.addHandler(file_handler)
        log.info(f"append file handler: {filename}")

max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 715
max_steps = 19073 # 19,073 steps is ~1 epoch, if data is 10B tokens and batch size 0.5M tokens
# for testing
warmup_iters = 20 # warm up 375e6 tokens, 375e6 / 0.5B(one epoch) = 715
max_steps = 1000 # if train 10e9, 10e9 / 2**19(one epoch) = 200000

def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    # 2) if it > lr_decay_iters, return min learning rate
    if it > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (
        1.0 + math.cos(math.pi * decay_ratio)
    )  # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)

def train(
    model,
    optimizer,
    lr_scheduler,
    dataloader,
    train_config:TrainConfig
):
    timer = create_timer(True)
    for step in range(train_config.max_steps):
        timer('start')
        t0 = time.time()
        last_step = step == train_config.max_steps - 1

        model.train()
        optimizer.zero_grad()
        loss_accum = 0.0
        timer(f'micro start - {train_config.grad_accum_steps}')
        for micro_step in range(train_config.grad_accum_steps):
            timer('next batch')
            x, y = dataloader.next_batch()
            x, y = x.to(train_config.device), y.to(train_config.device)
            # added after video, this field is also used by the forward pass.
            if train_config.ddp:
                model.require_backward_grad_sync = micro_step == train_config.grad_accum_steps - 1
            timer('forward')
            with torch.autocast(device_type=train_config.device, dtype=torch.bfloat16):
                logits, loss = model(x, y)
            # we have to scale the loss to account for gradient accumulation,
            # because the gradients just add on each successive backward().
            # addition of gradients corresponds to a SUM in the objective, but
            # instead of a SUM we want MEAN. Scale the loss here so it comes out right
            loss = loss / train_config.grad_accum_steps
            loss_accum += loss.detach()
            timer('backward')
            loss.backward()
        timer('optimizer start')
        if train_config.ddp:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # determine and set the learning rate for this iteration
        lr = lr_scheduler(step)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        optimizer.step()
        timer('logging')
        if train_config.device == "cuda":
            torch.cuda.synchronize()  # wait for the GPU to finish work
        t1 = time.time()
        dt = t1 - t0  # time difference in seconds
        tokens_processed = (
            dataloader.B * dataloader.T 
            * train_config.grad_accum_steps 
            * train_config.ddp_world_size
        )
        tokens_per_sec = tokens_processed / dt
        if train_config.ddp_master:
            print(
                f"step {step:5d} | loss: {loss_accum.item():.6f} | lr {lr:.4e} | norm: {norm:.4f} | dt: {dt * 1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}"
            )

def run(**args):
    # load config with overrided values
    config = get_config(**args)
    log.info(f"config : {config}")

    # logger setting
    append_file_handler(log, config)
    log.info(f"update logger args: {args}")

    # load dataset
    train_loader = DataLoaderLite(
        config.batch_size,
        config.sequence_length,
        config.local_rank,
        config.num_processes,
        "train",
    )
    val_loader = DataLoaderLite(
        config.batch_size,
        config.sequence_length,
        config.local_rank,
        config.num_processes,
        "val",
    )
    torch.set_float32_matmul_precision('high')

    # load model
    model, raw_model = load_model(config)

    # load optimizer
    optimizer = raw_model.configure_optimizers(
        weight_decay=0.1, learning_rate=6e-4, device=config.device
    )

    # run train (always with validation)
    train_config = from_dict(TrainConfig, config)
    train(
        model,
        optimizer,
        get_lr,
        train_loader,
        train_config
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="run.py", description="Train own GPT based on preset config"
    )

    model_group = parser.add_argument_group("Model")
    model_group.add_argument(
        "-p", "--preset", type=str, default="gpt2", help="gpt2,gpt2-xl"
    )

    train_group = parser.add_argument_group("Train")
    train_group.add_argument(
        "-d", "--device", type=str, default=None, help="gpt2,gpt2-xl"
    )

    # parser.add_argument("-lr", "--learning_rate", type=float)
    # parser.add_argument("--max_iters", type=int, default=1000)
    # parser.add_argument("--device", type=str, default="cuda")


    opt = parser.parse_args()
    arg_dict = vars(opt)
    defined_args = {k: v for k, v in arg_dict.items() if v is not None}
    log.info(f"initial args: {defined_args}")
    run(**defined_args)
