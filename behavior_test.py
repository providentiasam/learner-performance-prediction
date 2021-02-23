import argparse
from bt_case_question_prior import gen_question_prior
import pandas as pd
import torch
import torch.nn as nn

import pickle
from train_utils import get_data, get_chunked_data, prepare_batches, eval_batches
from train_lightning import DataModule, predict_saint
from models.model_lightning import SAKT, SAINT, DKT

from bt_case_perturbation import (
    gen_perturbation, test_perturbation,
    perturb_insertion_random, perturb_delete_random, perturb_replace_random,
)
from bt_case_original import test_original
from bt_case_repetition import gen_repeated_feed, test_repeated_feed
from bt_case_question_prior import gen_question_prior, test_question_prior
from bt_case_continuity import gen_continuity, test_continuity
from utils import *
import pytorch_lightning as pl
from sklearn.metrics import roc_auc_score, accuracy_score

SUMMARY_PATH = './summary_original_saint.csv'

if __name__ == "__main__":
    if not os.path.exists(SUMMARY_PATH):
        pd.DataFrame().to_csv(SUMMARY_PATH)
    summary_csv = pd.read_csv(SUMMARY_PATH, index_col=0)
    print(summary_csv),

    parser = argparse.ArgumentParser(description="Behavioral Testing")
    parser.add_argument("--dataset", type=str, default="ednet")
    parser.add_argument("--model", type=str, \
        choices=["lr", "dkt_legacy", "dkt1", "sakt_legacy",\
             "sakt", "saint", "dkt"], default="sakt")
    parser.add_argument("--test_type", type=str, default="original")
    parser.add_argument("--load_dir", type=str, default="./save/")
    parser.add_argument("--filename", type=str, default="best")
    parser.add_argument("--gpu", type=str, default="4,5,6,7")
    parser.add_argument("--diff_threshold", type=float, default=0)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # 1. LOAD PRE-TRAINED MODEL + EXP ARGS - {SAINT, DKT, BESTLR, SAKT}
    if args.model in {'saint', 'sakt', 'dkt'}:
        checkpoint_path = f'./save/{args.model}/{args.dataset}/' + args.filename + '.ckpt'
        with open(checkpoint_path.replace('.ckpt', '_config.pkl'), 'rb') as file:
            model_config = argparse.Namespace(**pickle.load(file))
        print(model_config)
        model_config.project = 'bt_bt'
        model_config.use_wandb = False
        if args.model.startswith('saint'):
            model = SAINT.load_from_checkpoint(checkpoint_path, config=model_config).cuda()
        elif args.model.startswith('sakt'):
            model = SAKT.load_from_checkpoint(checkpoint_path, config=model_config).cuda()
        elif args.model.startswith('dkt'):
            model = DKT.load_from_checkpoint(checkpoint_path, config=model_config).cuda()
        model_seq_len = vars(model.config)['seq_len']
        model.eval()
    else:
        print(args.load_dir + f'/{args.model}/{args.dataset}/{args.filename}')
        saver = Saver(args.load_dir + f'/{args.model}/{args.dataset}/', args.filename)
        model = saver.load().cuda()
        model.eval()
        model_seq_len = {'statics': 200, 'spanish': 200, \
            'ednet_small': 100, 'assistments15': 100, 'assistments17': 100,\
                'ednet_medium': 100, 'ednet': 100}[args.dataset]
        model_config = argparse.Namespace(**{})

    if args.test_type == 'original':
        model_seq_len = None

    test_kwargs = {}
    if args.test_type in ['reconstruction', 'repetition']:
        test_kwargs['item_or_skill'] = 'item'
    if args.test_type in ['insertion', 'deletion', 'replacement']:
        test_kwargs['insert_policy'] = 'middle'
        test_kwargs['perturb_func'] = {
            'insertion': perturb_insertion_random,
            'deletion': perturb_delete_random,
            'replacement': perturb_replace_random,
        }[args.test_type]
    last_one_only = {
        'reconstruction': True, 'repetition': False, 'insertion': False,
        'deletion': False, 'replacement': False, 'original': False, 
        'question_prior': False, 'continuity': False
    }[args.test_type]

    # 2. GENERATE TEST DATA.
    # bt_test_df: generated test dataset for behavioral testing
    # other_info: any meta information stored w.r.t the test case
    bt_test_path = os.path.join("data", args.dataset, "bt_{}.pkl".format(args.test_type))
    USE_CACHE = True
    if os.path.exists(bt_test_path) and USE_CACHE:
        print("Loading existing bt test data file.")
        with open(bt_test_path, 'rb') as file:
            bt_test_df, other_info = pickle.load(file)
    else:
        test_df = pd.read_csv(
            os.path.join("data", args.dataset, "preprocessed_data_test.csv"), sep="\t"
        ) # Original Test-split DataFrame
        print('Data Loaded', test_df.shape[0])
        if model_seq_len is not None:
            test_df = test_df.groupby('user_id').head(model_seq_len).reset_index(drop=True)
        gen_funcs = {
            'repetition': gen_repeated_feed,
            'insertion': gen_perturbation,
            'deletion': gen_perturbation,
            'replacement': gen_perturbation,
            'question_prior': gen_question_prior,
            'continuity': gen_continuity,
            'original': lambda x: (x, None)
        }
        bt_test_df, other_info = gen_funcs[args.test_type](test_df, **test_kwargs)
        if args.test_type != 'original' and USE_CACHE:
            with open(bt_test_path, 'wb+') as file:
                pickle.dump((bt_test_df, other_info), file)
        del test_df


    # 3. FEED TEST DATA.
    # In: bt_test_df
    # Out: bt_test_df with 'model_pred' column.
    if args.model in {'saint', 'sakt', 'dkt'}:
        datamodule = DataModule(model_config, overwrite_test_df=bt_test_df, \
            last_one_only=last_one_only, overwrite_test_batch=3000)
        trainer = pl.Trainer(
            gpus=len(args.gpu.split(',')), accelerator='dp',
            auto_select_gpus=True, callbacks=[], max_steps=0)
        trainer.test(model=model, datamodule=datamodule)
        bt_test_preds = model.preds[0]
        bt_test_labels = model.labels[0]
        if last_one_only:
            bt_test_df = bt_test_df.groupby('user_id').last()
        bt_test_df['model_pred'] = bt_test_preds.cpu()
    else:
        if args.model == 'sakt_legacy' or (args.model =='dkt' and args.dataset == 'ednet'): 
            bt_test_data, _ = get_chunked_data(bt_test_df, max_length=500, \
                train_split=1.0, stride=1)
        else:
            bt_test_data, _ = get_data(bt_test_df, train_split=1.0, randomize=False, model_name=args.model)
        bt_test_batch = prepare_batches(bt_test_data, 128, False)
        bt_test_preds = eval_batches(model, bt_test_batch, 'cuda', model_name=args.model)
        bt_test_df['model_pred'] = bt_test_preds
        if last_one_only:
            bt_test_df = bt_test_df.groupby('user_id').last()


    # 4. CHECK PASS CONDITION AND RUN CASE-SPECIFIC ANALYSIS.
    test_funcs = {
        'repetition': test_repeated_feed,
        'insertion': lambda x: test_perturbation(x, diff_threshold=args.diff_threshold),
        'deletion': lambda x: test_perturbation(x, diff_threshold=args.diff_threshold),
        'replacement': lambda x: test_perturbation(x, diff_threshold=args.diff_threshold),
        'question_prior': lambda x: test_question_prior(x, item_meta=other_info, test_name=args.model),
        'continuity': test_continuity,
        'original': test_original
    }
    result_df, summary_df = test_funcs[args.test_type](bt_test_df)
        # result_df: bt_test_df appended with 'test_measure' column or any additional test-case-specific info.
        # groupby_key: list of column names in result_df for 5's mutually exclusive subset-wise analysis.


    # 5. GET SUMMARY STAT.
    result_dict = {}
    eval_col = 'test_measure'
    result_df['all'] = 'all'
    result_df_path = f'./results/{args.dataset}_{args.test_type}_{args.model}.pkl'
    with open(result_df_path, 'wb+') as file:
        pickle.dump(result_df, file)
    summary_df.to_csv(result_df_path.replace('.pkl', '_summary.csv'))
    # summary content: (grouped) auc, acc, time

    #6. APPEND SUMMARY.
    # time, model, dataset, test type
    summary_df['model'] = args.model
    summary_df['dataset'] = args.dataset
    summary_df['testtype'] = args.test_type
    summary_df['time'] = str(pd.datetime.now()).split('.')[0]
    summary_df = summary_df.reset_index(drop=False)
    new_summary = pd.concat([summary_csv, summary_df], axis=0).reset_index(drop=True)
    new_summary.to_csv(SUMMARY_PATH)
    print(summary_df)