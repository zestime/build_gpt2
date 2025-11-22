import time
import logging
import argparse

from src.checkpoint_manager import CheckPointManager
import tiktoken
from config import get_config, PicoGPTConfig, TrainConfig, from_dict
from dataset import DataLoaderLite
from model import load_model
import torch
import torch.distributed as dist
import math
from torch.nn import functional as F
from pathlib import Path

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
        logging_dir = Path(f"{config.logging_dir_prefix}/{config.execution_id}")
        logging_dir.mkdir(parents=True, exist_ok=True)
        filename = logging_dir / "train.log"
        file_handler = logging.FileHandler(filename, "w")
        file_handler.setLevel(logging.INFO)
        logger.addHandler(file_handler)
        log.info(f"append file handler: {filename}")

def create_lr_scheduler(config:PicoGPTConfig):
    max_lr = config.max_learning_rate
    min_lr = config.min_learning_rate
    warmup_steps = config.warmup_steps
    max_steps = config.max_steps

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
    return get_lr

def eval(model, val_loader, train_config:TrainConfig):
    model.eval()
    val_loader.reset()
    with torch.no_grad():
        val_loss_accum = 0.0
        val_loss_steps = 20
        for _ in range(val_loss_steps):
            x, y = val_loader.next_batch()
            x, y = x.to(train_config.device), y.to(train_config.device)
            with torch.autocast(device_type=train_config.device, dtype=torch.bfloat16):
                logits, loss = model(x, y)
            loss = loss / val_loss_steps
            val_loss_accum += loss.detach()
    if train_config.ddp:
        dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
    if train_config.ddp_master:
        log.info(f"validation loss: {val_loss_accum.item():.4f}")

def inference(model, tokenizer, train_config:TrainConfig):
    model.eval()
    num_return_sequences = 8
    max_length = 32
    tokens = tokenizer.encode("Hello, I am a language model,")
    tokens = torch.tensor(tokens, dtype=torch.long)
    tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
    xgen = tokens.to(train_config.device)
    sample_rng = torch.Generator(device=train_config.device)
    sample_rng.manual_seed(42 + train_config.ddp_rank)
    while xgen.size(1) < max_length:
        # forward the model to get the logits
        with torch.no_grad():
            with torch.autocast(device_type=train_config.device, dtype=torch.bfloat16):
                logits, loss = model(xgen) # (B, T, vocab_size)
            # take the logits at the last position
            logits = logits[:, -1, :] # (B, vocab_size)
            # get the probabilities
            probs = F.softmax(logits, dim=-1)
            # do top-k sampling of 50 (huggingface pipeline default)
            # topk_probs here becomes (5, 50), topk_indices is (5, 50)
            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
            # select a token from the top-k probabilities
            # note: multinomial does not demand the input to sum to 1
            ix = torch.multinomial(topk_probs, 1, generator=sample_rng) # (B, 1)
            # gather the corresponding indices
            xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
            # append to the sequence
            xgen = torch.cat((xgen, xcol), dim=1)
    # print the generated text
    for i in range(num_return_sequences):
        tokens = xgen[i, :max_length].tolist()
        decoded = tokenizer.decode(tokens)
        print(f"rank {train_config.ddp_rank} sample {i}: {decoded}")
    

def train(
    tokenizer,
    model,
    optimizer,
    lr_scheduler,
    train_loader,
    val_loader,
    train_config:TrainConfig
):
    checkpoint_manager = CheckPointManager(
        f"{train_config.logging_dir_prefix}/{train_config.execution_id}",
        train_config.metric_name,
        train_config.max_to_keep,
        train_config.mode
    )

    for step in range(1, train_config.max_steps + 1):
        last_step = step == train_config.max_steps + 1

        if train_config.eval_enable and step % train_config.eval_interval == 0:
            eval(
                model,
                val_loader,
                train_config
            )

        if train_config.infer_enable and step % train_config.infer_interval == 0:
            inference(
                model,
                tokenizer,
                train_config
            )


        t0 = time.time()
        model.train()
        optimizer.zero_grad()
        loss_accum = 0.0
        for micro_step in range(train_config.grad_accum_steps):
            x, y = train_loader.next_batch()
            x, y = x.to(train_config.device), y.to(train_config.device)
            # added after video, this field is also used by the forward pass.
            if train_config.ddp:
                model.require_backward_grad_sync = micro_step == train_config.grad_accum_steps - 1
            with torch.autocast(device_type=train_config.device, dtype=torch.bfloat16):
                logits, loss = model(x, y)
            # we have to scale the loss to account for gradient accumulation,
            # because the gradients just add on each successive backward().
            # addition of gradients corresponds to a SUM in the objective, but
            # instead of a SUM we want MEAN. Scale the loss here so it comes out right
            loss = loss / train_config.grad_accum_steps
            loss_accum += loss.detach()
            loss.backward()
        if train_config.ddp:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # determine and set the learning rate for this iteration
        lr = lr_scheduler(step)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        optimizer.step()
        if train_config.device == "cuda":
            torch.cuda.synchronize()  # wait for the GPU to finish work
        t1 = time.time()
        dt = t1 - t0  # time difference in seconds
        tokens_processed = (
            train_loader.B * train_loader.T 
            * train_config.grad_accum_steps 
            * train_config.ddp_world_size
        )
        tokens_per_sec = tokens_processed / dt
        if train_config.ddp_master:
            log.info(
                f"step {step:5d} | loss: {loss_accum.item():.6f} | lr {lr:.4e} | norm: {norm:.4f} | dt: {dt * 1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}"
            )

        if train_config.checkpoint_interval > 0 and (
            step % train_config.checkpoint_interval == 0 or last_step
        ):
            checkpoint_manager.save_checkpoint(
                step,
                model,
                optimizer,
                metric_value=loss_accum.item(),
                additional_info=None
            )

def run(**args):
    # load config with overrided values
    config = get_config(**args)

    if (config.preset == "debug"):
        log.debug("debug mode")
        log.debug(f"args: {args}")
        log.info(f"config : {config}")

    # logger setting
    append_file_handler(log, config)
    log.info(f"update logger args: {args}")

    tokenizer = tiktoken.get_encoding("gpt2")

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


    lr_scheduler = create_lr_scheduler(
        config
    )

    # run train (always with validation)
    train_config = from_dict(TrainConfig, config)
    train(
        tokenizer,
        model,
        optimizer,
        lr_scheduler,
        train_loader,
        val_loader,
        train_config
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="run.py", description="Train own GPT based on preset config"
    )

    model_group = parser.add_argument_group("Model")
    model_group.add_argument(
        "-p", "--preset", type=str, default="test", help="debug,test,gpt2,gpt2-xl"
    )

    model_group.add_argument(
        "-pos", "--position", type=str, default="ape", help="ape,rope"
    )

    train_group = parser.add_argument_group("Train")
    train_group.add_argument(
        "-ei", "--infer_interval", type=int, default=None, help="500"
    )
    train_group.add_argument(
        "-ev", "--eval_interval", type=int, default=None, help="1000"
    )
    train_group.add_argument(
        "-d", "--device", type=str, default=None, help="cpu,cuda"
    )
    train_group.add_argument(
        "--rope", action="store_true", help="use RoPE"
    )
    train_group.add_argument(
        "--ape", action="store_true", help="use APE(absolute position encoding)"
    )

    train_group.add_argument("--max_steps", type=int, default=None)

    opt = parser.parse_args()
    arg_dict = vars(opt)
    defined_args = {k: v for k, v in arg_dict.items() if v is not None}
    log.info(f"initial args: {defined_args}")
    run(**defined_args)
