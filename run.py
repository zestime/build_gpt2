import argparse
from config import GPTConfig
from preset import GPT2Preset, GPT2XLConfig
from util import start_interactive_shell


def get_preset(preset: str) -> GPTConfig:
    match preset:
        case "gpt2":
            return GPT2Preset
        case "gpt2-xl":
            return GPT2XLConfig
        case _:
            raise ValueError(f"Unknown preset: {preset}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="run.py",
        description="Train own GPT based on preset config")

    model_group = parser.add_argument_group("Model")
    model_group.add_argument("-p", "--preset", type=str, default='gpt2', help='gpt2,gpt2-xl')
    train_group = parser.add_argument_group("Train")
    train_group.add_argument("-d", "--device", type=str, default='cuda', help='gpt2,gpt2-xl')

    # parser.add_argument("-lr", "--learning_rate", type=float)
    # parser.add_argument("--max_iters", type=int, default=1000)
    # parser.add_argument("--device", type=str, default="cuda")
    
    opt = parser.parse_args()
    print(f"opt: {opt}")
    arg_dict = vars(opt)
    # defined_args = {k: v for k, v in arg_dict.items() 
    #                 if v is not None or k != "preset"}
    # config = get_preset(**defined_args)
    # print(config)

    start_interactive_shell(locals())

    