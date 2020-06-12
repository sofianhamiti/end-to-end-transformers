import os
import argparse

def str2bool(v): ## added for SageMaker run
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

        
def parse_args():
    parser = argparse.ArgumentParser()

    # data, model, and output directories
    parser.add_argument("--train", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    parser.add_argument("--validation", type=str, default=os.environ["SM_CHANNEL_VALIDATION"])
    parser.add_argument("--output_dir", type=str, default=os.environ["SM_MODEL_DIR"])
    # model parameters
    parser.add_argument("--model_name_or_path", default=None, type=str)
    parser.add_argument("--task", default=None, type=str)
    # other parameters
    parser.add_argument("--do_train", type=str2bool, nargs='?', const=True, default=False) # modified for SageMaker use
    parser.add_argument("--do_predict", type=str2bool, nargs='?', const=True, default=False) # modified for SageMaker use
    parser.add_argument("--n_gpu", type=int, default=int(os.environ["SM_NUM_GPUS"]))
    parser.add_argument("--config_name", default="", type=str)
    parser.add_argument("--tokenizer_name", default="", type=str)
    parser.add_argument("--cache_dir", default="", type=str)
    parser.add_argument("--max_seq_length", default=None, type=int)
    parser.add_argument("--train_batch_size", default=32, type=int)
    parser.add_argument("--eval_batch_size", default=32, type=int)
    parser.add_argument('--gradient_accumulation_steps', default=1, type=int)     
    parser.add_argument("--learning_rate", default=5e-5, type=float)
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--num_train_epochs", default=3, type=float)
    parser.add_argument("--max_steps", default=-1, type=int)
    parser.add_argument("--warmup_steps", default=0, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--fp16_opt_level', default='O1', type=str)

    args, _ = parser.parse_known_args()
    return args