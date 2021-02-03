import argparse
from bt_case_question_prior import gen_question_prior
import pandas as pd
import torch

from models.model_dkt2 import DKT2
from models.model_sakt2 import SAKT

import pickle
from train_utils import get_data, get_chunked_data, prepare_batches, eval_batches
from train_saint import SAINT, DataModule, predict_saint

from bt_case_perturbation import (
    gen_perturbation, test_perturbation,
    perturb_insertion_random, perturb_delete_random, perturb_replace_random,
)
from bt_case_reconstruction import gen_knowledge_state, test_knowledge_state, test_simple
from bt_case_repetition import gen_repeated_feed
from bt_case_question_prior import gen_question_prior, test_question_prior
from utils import *
import pytorch_lightning as pl
from sklearn.metrics import roc_auc_score, accuracy_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Behavioral Testing")
    parser.add_argument("--dataset", type=str, default="spanish")
    parser.add_argument("--model", type=str, \
        choices=["lr", "dkt", "dkt1", "sakt", "saint"], default="saint")
    parser.add_argument("--test_type", type=str, default="insertion")
    parser.add_argument("--load_dir", type=str, default="./save/")
    parser.add_argument("--filename", type=str,\
         default="spanish")
    parser.add_argument("--gpu", type=str, default="1,2,3")
    parser.add_argument("--diff_threshold", type=float, default=0)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # 1. LOAD DATA + PRE-TRAINED MODEL - {SAINT, DKT, BESTLR, SAKT}
    if args.model == 'saint':
        checkpoint_path = f'./save/{args.model}/' + args.filename + '.ckpt'
        with open(checkpoint_path.replace('.ckpt', '_config.pkl'), 'rb') as file:
            model_config = argparse.Namespace(**pickle.load(file))
        print(model_config)
        model = SAINT.load_from_checkpoint(checkpoint_path, config=model_config\
            ).to(torch.device("cuda"))
        model.eval()
    else:
        print(args.load_dir + f'/{args.model}/{args.dataset}/{args.filename}')
        saver = Saver(args.load_dir + f'/{args.model}/{args.dataset}/', args.filename)
        model = saver.load().to(torch.device("cuda"))
        model.eval()
        model_config = argparse.Namespace(**{})

    test_df = pd.read_csv(
        os.path.join("data", args.dataset, "preprocessed_data_test.csv"), sep="\t"
    ) # Original Test-split DataFrame

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
        'question_prior': False
    }[args.test_type]


    # 2. GENERATE TEST DATA.
    gen_funcs = {
        'reconstruction': gen_knowledge_state,
        'repetition': gen_repeated_feed,
        'insertion': gen_perturbation,
        'deletion': gen_perturbation,
        'replacement': gen_perturbation,
        'question_prior': gen_question_prior,
        'original': lambda x: (x, None)
    }
    bt_test_df, other_info = gen_funcs[args.test_type](test_df, **test_kwargs)
        # bt_test_df: generated test dataset for behavioral testing
        # other_info: any meta information stored w.r.t the test cas6e


    # 3. FEED TEST DATA.
    # In: bt_test_df
    # Out: bt_test_df with 'model_pred' column.
    bt_test_path = os.path.join("data", args.dataset, "bt_{}.csv".format(args.test_type))
    original_test_df = bt_test_df.copy()
    original_test_df.to_csv(bt_test_path)
    if args.model == 'saint':
        datamodule = DataModule(model_config, overwrite_test_df=bt_test_df, \
            last_one_only=last_one_only)
        trainer = pl.Trainer(auto_select_gpus=True, callbacks=[], max_steps=0)
        bt_test_preds = predict_saint(saint_model=model, dataloader=datamodule.test_dataloader())
        if last_one_only:
            bt_test_df = bt_test_df.groupby('user_id').last()
        bt_test_df['model_pred'] = bt_test_preds.cpu()
    else:
        if args.model == 'sakt': 
            bt_test_data, _ = get_chunked_data(bt_test_df, max_length=200, \
                train_split=1.0, stride=10)
        else:
            bt_test_data, _ = get_data(bt_test_df, train_split=1.0, randomize=False)
        bt_test_batch = prepare_batches(bt_test_data, 1, False)
        bt_test_preds = eval_batches(model, bt_test_batch, 'cuda', args.model == 'dkt1')
        print(len(bt_test_df))
        bt_test_df['model_pred'] = bt_test_preds
        if last_one_only:
            bt_test_df = bt_test_df.groupby('user_id').last()


    # 4. CHECK PASS CONDITION AND RUN CASE-SPECIFIC ANALYSIS.
    test_funcs = {
        'reconstruction': test_knowledge_state,
        'repetition': test_simple,
        'insertion': lambda x: test_perturbation(x, diff_threshold=args.diff_threshold),
        'deletion': lambda x: test_perturbation(x, diff_threshold=args.diff_threshold),
        'replacement': lambda x: test_perturbation(x, diff_threshold=args.diff_threshold),
        'question_prior': lambda x: test_question_prior(x, item_meta=other_info, test_name=args.model),
        'original': lambda x: test_simple(x, testcol='correct')
    }
    result_df, groupby_key = test_funcs[args.test_type](bt_test_df)
        # result_df: bt_test_df appended with 'testpass' column or any additional test-case-specific info.
        # groupby_key: list of column names in result_df for 5's mutually exclusive subset-wise analysis.


    # 5. GET SUMMARY STAT.
    result_dict = {}
    eval_col = 'testpass'
    result_df['all'] = 'all'
    for group_key in groupby_key:
        result_dict[group_key] = result_df.groupby(group_key)[eval_col].describe()
    metric_df = pd.concat([y for _, y in result_dict.items()], axis=0, keys=result_dict.keys())
    print(metric_df)
    print("auc_test = ", roc_auc_score(result_df["correct"], result_df['model_pred']))
    result_df.to_csv(f'./results/{args.dataset}_{args.test_type}_{args.model}.csv')
