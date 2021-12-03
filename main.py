
import os
from argparse import ArgumentParser
import pytorch_lightning as pl

from q_snippets.object import Config, print_config
from omegaconf import OmegaConf as oc

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = FILE_DIR[:FILE_DIR.index('src')]

from frame import train_model, run_prediction

def CLI_parser():
    parser = ArgumentParser()
    parser.add_argument("--mode", type=str, default='train', choices=['train','predict','resume', 'eval', 'cv', 'ensemble'])
    parser.add_argument("--config", type=str, default=None, help="recover from config file,  since config is used, all modification will have NO effect")
    parser.add_argument("--KFold", type=int, default=None, help="KFold training")
    parser.add_argument("--rand_seed", type=int, default=12345)
    parser.add_argument("--model", type=str, default='mrc')

    # trainer
    parser.add_argument("--version", type=str, default='tmp')
    parser.add_argument("--accumulate_grads", type=int, default=1, help="accumulate_grads")
    parser.add_argument("--max_epochs", type=int, default=20, help="stop training when reaches max_epochs")
    
    # data
    parser.add_argument("--dataset", type=str, default=None, dest='data.dataset')  
    parser.add_argument("--train_path", type=str, default=None, dest="data.train_path")
    parser.add_argument("--val_path", type=str, default=None, dest="data.val_path")
    parser.add_argument("--test_path", type=str, default=None, dest="data.test_path")
    parser.add_argument("--train_bsz", type=int, default=8, dest="data.train_bsz")
    parser.add_argument("--val_bsz", type=int, default=8, dest="data.val_bsz")
    parser.add_argument("--test_bsz", type=int, default=16, dest="data.test_bsz")
    parser.add_argument("--cache_dir", type=str, default='./cache', help="cache path for dataset", dest="data.cache_dir")

    # model
    # 本地无此文件则使用 "hfl/chinese-roberta-wwm-ext-large", huggingface 会自动下载
    parser.add_argument("--pretrained", type=str, default="/pretrains/pt/hfl-chinese-roberta-wwm-ext-large") 
    parser.add_argument("--adversarial", action='store_true', default=False)
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--dropout", type=float, default=0.1, help="encoder dropout rate")
    parser.add_argument("--loss_fn", type=str, default='ce', help="loss function in model implementation")

    # test / prediction
    parser.add_argument("--ckpt_path", type=str, default=None, help="saved ckpt path for testing/prediction", dest="preds.ckpt_path")
    parser.add_argument("--result_path", type=str, default="result.txt", help="result file to save", dest="preds.result_path")
    parser.add_argument("--nbest", type=int, default=1, help="if nbest > 1, do nbest prediction", dest="preds.nbest")
    parser.add_argument("--save_pg", action='store_true', default=False, help="if set, gold and pred will be saved for contrast", dest="preds.save_pg")

    parser.add_argument("--eval_script", type=str, default=None)
    return parser

if __name__ == "__main__":
    
    parser = CLI_parser()
    args = parser.parse_args()
    pl.seed_everything(args.rand_seed)

    settings = {
        'mode': 'train',
        'pretrained': "/pretrains/pt/bart_nl2sparql",
        # 'pretrained': "/pretrains/pt/hfl-chinese-bert-wwm",
        'model': 'raw',
        'accumulate_grads': 32,
        'KFold': 5,
        'lr': 2e-5,
        'data': {
                'dataset': 'nl2sp',
                'tokenizer': "/pretrains/pt/bart_nl2sparql",
                'train_path': os.path.join(PROJ_DIR, "data/kqa/train.json"),
                'val_path': os.path.join(PROJ_DIR, "data/kqa/val.json"),
                'test_path': os.path.join(PROJ_DIR, "data/kqa/test.json"),
                'train_bsz': 4,
                'val_bsz': 4,
                'test_bsz': 4,
                'cache_dir': './cached/main/qa',
                'force_reload': False
            },
        'preds' : {
            'ckpt_path': "/home/transsion/projects/kbqa/src/query_cls/lightning_logs/tmp/epoch=18_val_f1=0.000.ckpt",
            'result_path': "./prediction.json",
        }
    }
    provided, default = Config.from_argparse(parser)
    config = oc.merge(default, oc.create(settings), provided)  # 优先级右边高 
    print_config(config)
    print(f"{provided} Overwrited by command line!")

    if config.mode == 'train':
        train_model(config)
    elif config.mode == 'predict':
        run_prediction(config)
    # elif config.mode == 'eval':
    #     run_eval(config)
    # elif config.mode == 'cv':
    #     torch.multiprocessing.set_start_method("spawn")
    #     cross_validation(config)
    # elif config.mode == 'ensemble':

    #     ckpts = [

    #     ]
    #     ensemble(config, ckpts)












