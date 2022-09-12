
from ensurepip import version
import os
from argparse import ArgumentParser
import pytorch_lightning as pl

from q_snippets.object import Config, print_config
from omegaconf import OmegaConf as oc

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = os.path.abspath("..")
print(f"proj_dir is: {PROJ_DIR}")

 
from frame import raw_generate, train_model, predict_ckpt

def CLI_parser():
    parser = ArgumentParser()
    parser.add_argument("--mode", type=str, default='train',  )
    parser.add_argument("--config", type=str, default=None, help="recover from config file, if this config is used then all modification will have NO effect")
    # parser.add_argument("--KFold", type=int, default=None, help="KFold training")
    parser.add_argument("--rand_seed", type=int, default=12345)

    # model
    # 本地无此文件则使用 "hfl/chinese-roberta-wwm-ext-large", huggingface 会自动下载
    parser.add_argument("--pretrained", type=str, default="/pretrains/pt/hfl-chinese-roberta-wwm-ext-large") 
    parser.add_argument("--model_name", type=str, default='simple')

    # trainer
    parser.add_argument("--version", type=str, default='tmp')
    parser.add_argument("--accumulate_grads", type=int, default=1, help="accumulate_grads")
    parser.add_argument("--max_epochs", type=int, default=20, help="stop training when reaches max_epochs")
    
    # data
    parser.add_argument("--dataset", type=str, default=None, dest='data.dataset', help="how to process data")  
    parser.add_argument("--train_path", type=str, default=None, dest="data.train_path")
    parser.add_argument("--val_path", type=str, default=None, dest="data.val_path")
    parser.add_argument("--test_path", type=str, default=None, dest="data.test_path")
    parser.add_argument("--train_bsz", type=int, default=8, dest="data.train_bsz")
    parser.add_argument("--val_bsz", type=int, default=8, dest="data.val_bsz")
    parser.add_argument("--test_bsz", type=int, default=16, dest="data.test_bsz")

    parser.add_argument("--cache_dir", type=str, default='./cache', help="cache path for dataset", dest="data.cache_dir")
    parser.add_argument("--tokenizer", type=str, dest="data.tokenizer", help="rarely used for pretrained path contains tokenizer")
    parser.add_argument("--force_reload", action='store_true', default=False, dest='data.force_reload')


    # test / prediction
    parser.add_argument("--ckpt_path", type=str, default=None, help="saved ckpt path for testing/prediction", dest="preds.ckpt_path")
    parser.add_argument("--result_path", type=str, default="result.txt", help="result file to save", dest="preds.result_path")


    parser.add_argument("--wandb", action='store_true', default=False)
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--warmup_rate", type=float, default=0.1, help="warmup_rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="weight_decay")
    return parser

if __name__ == "__main__":
    
    parser = CLI_parser()
    args = parser.parse_args()
    pl.seed_everything(args.rand_seed)

    settings = {
        # 'mode': 'predict',
        'mode': 'train',
        'version': "first",
        'model_name': 'bart',
        'pretrained': "/pretrains/pt/facebook-bart-base",
        'accumulate_grads': 1,
        'lr': 3e-5,
        'max_epochs' : 25,
        'data': {
                'dataset': 'seq2seq',
                'tokenizer': "/pretrains/pt/bart-base",
                # 'tokenizer': os.path.join(PROJ_DIR, "ckpt/bart-base"),
                'train_path': os.path.join(PROJ_DIR, "data/demo/train.json"),
                'val_path': os.path.join(PROJ_DIR, "data/demo/val.json"),
                'test_path': os.path.join(PROJ_DIR, "data/demo/val.json"),
                'train_bsz': 8,
                'val_bsz': 64,
                'test_bsz': 64,
                'cache_dir': './cached/main',
                'force_reload': False
            },
        'preds' : {

            'ckpt_path' : "lightning_logs/first/epoch=20_train_loss=0.0018.ckpt",      
            'result_path': "./results/first.json",
        }
    }
    provided, default = Config.from_argparse(parser)
    config = oc.merge(default, oc.create(settings), provided)  # 优先级右边高 
    print_config(config)
    print(f"{provided} Overwrited by command line!")

    if config.mode in ['train', 'resume']:
        train_model(config)
    elif config.mode == 'predict':
        predict_ckpt(config)
    elif config.mode == "generate":
        raw_generate(config)