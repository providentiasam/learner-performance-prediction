import argparse
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score

import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.nn.utils import clip_grad_norm_

from train_utils import *
from models.model_sakt2 import SAKT
from utils import *
import itertools


def train(train_data, val_data, model, optimizer, logger, saver, num_epochs, batch_size, grad_clip):
    """Train SAKT model.
    Arguments:
        train_data (list of tuples of torch Tensor)
        val_data (list of tuples of torch Tensor)
        model (torch Module)
        optimizer (torch optimizer)
        logger: wrapper for TensorboardX logger
        saver: wrapper for torch saving
        num_epochs (int): number of epochs to train for
        batch_size (int)
        grad_clip (float): max norm of the gradients
    """
    criterion = nn.BCEWithLogitsLoss()
    metrics = Metrics()
    step = 0

    if optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler = None
    elif optimizer == 'noam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        def noam(step: int):
            step = max(1, step)
            warmup_steps = 2000
            scale = warmup_steps ** 0.5 * min(
                step ** (-0.5), step * warmup_steps ** (-1.5))
            return scale
        scheduler = LambdaLR(optimizer=optimizer, lr_lambda=noam)
    else:
        raise NotImplementedError

    val_batches = prepare_batches(val_data, batch_size, randomize=False)

    for epoch in range(num_epochs):
        train_batches = prepare_batches(train_data, batch_size)
        # Training
        for item_inputs, skill_inputs, label_inputs, item_ids, skill_ids, labels in train_batches:
            item_inputs = item_inputs.cuda()
            skill_inputs = skill_inputs.cuda()
            label_inputs = label_inputs.cuda()
            item_ids = item_ids.cuda()
            skill_ids = skill_ids.cuda()

            preds = model(item_inputs, skill_inputs, label_inputs, item_ids, skill_ids)
            loss = compute_loss(preds, labels.cuda(), criterion)
            preds = torch.sigmoid(preds).detach().cpu()
            train_auc = compute_auc(preds, labels)

            model.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            step += 1
            metrics.store({'loss/train': loss.item()})
            metrics.store({'auc/train': train_auc})
            if scheduler is not None:
                metrics.store({'lr': scheduler.get_last_lr()[0]})
                scheduler.step()
            # Logging
            if step % 20 == 0:
                logger.log_scalars(metrics.average(), step)

        # Validation
        model.eval()
        for item_inputs, skill_inputs, label_inputs, item_ids, skill_ids, labels in val_batches:
            item_inputs = item_inputs.cuda()
            skill_inputs = skill_inputs.cuda()
            label_inputs = label_inputs.cuda()
            item_ids = item_ids.cuda()
            skill_ids = skill_ids.cuda()
            with torch.no_grad():
                preds = model(item_inputs, skill_inputs, label_inputs, item_ids, skill_ids)
                preds = torch.sigmoid(preds).cpu()
            val_auc = compute_auc(preds, labels)
            metrics.store({'auc/val': val_auc})
        model.train()

        # Save model
        average_metrics = metrics.average()
        average_metrics['epoch'] = epoch
        logger.log_scalars(average_metrics, step)
        stop = saver.save(average_metrics['auc/val'], model, epoch, average_metrics)
        if stop:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train SAKT.')
    parser.add_argument('--setup', type=str or bool, default='ednet_small')
    parser.add_argument('--gpus', type=str, default='4,5,6,7')
    args_ = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args_.gpus
    DEBUGGING = True
    REPEAT = 1
    if DEBUGGING:
        setup_path = './setups/sakt_loop_test.xlsx'
        setup_page = pd.DataFrame([{
            'dataset': 'ednet_small',
            'num_attn_layers': 2,
            'max_length': 100,
            'embed_size': 64,
            'num_heads': 16,
            'encode_pos': 1, 'max_pos': 10, 'drop_prob': 0.25, 'batch_size': 100, 'optimizer': 'noam',
            'lr': 0.003, 'grad_clip': 10, 'num_epochs': 5, 'repeat': 1, 'stride': 50,
            'query_feed': True, 'query_highpass': True
        }
        ])
    elif args_.setup:
        setup_path = './setups/sakt_loop_{}.xlsx'.format(args_.setup)
        setup_page = pd.read_excel(setup_path)
        if 'dataset' not in setup_page.columns:
            setup_page['dataset'] = args_.setup
    else:
        setup_path = './setups/sakt_{}.xlsx'.format(str(pd.datetime.now()).split('.')[0])
        setup_products = {
            'dataset': ['ednet_medium'],
            'num_attn_layers': [3],
            'max_length': [400],
            'embed_size': [64],
            'num_heads': [16],
            'drop_prob': [0.25],
            'optimizer': ['noam'],
            'lr': [0.003],
            'encode_pos': [1], 'max_pos': [10], 'batch_size': [75], 
            'grad_clip': [10], 'num_epochs': [100], 'repeat': [REPEAT], 'stride': [50],
            'query_feed': [False], 'query_highpass': [False]
        }
        setup_page = pd.DataFrame(list(itertools.product(*[y for _, y in setup_products.items()])))
        setup_page.columns = setup_products.keys()
  
    result_cols = [f'{x}{i+1}' for x in ['test', 'valid', 'train'] for i in range(REPEAT)] + ['remark']
    setup_cols = [x for x in setup_page.columns if x not in result_cols]
    for col in result_cols:
        if col not in setup_page.columns:
            setup_page[col] = np.nan    
    setup_page[setup_cols] = setup_page[setup_cols].ffill()

    save_path = setup_path
    loop_through = setup_page.index
    for setup_index in loop_through:
        try:
            dataset = setup_page['dataset'][setup_index]
            full_df = pd.read_csv(os.path.join('data', dataset, 'preprocessed_data.csv'), sep="\t")
            train_df = pd.read_csv(os.path.join('data', dataset, 'preprocessed_data_train.csv'), sep="\t")
            test_df = pd.read_csv(os.path.join('data', dataset, 'preprocessed_data_test.csv'), sep="\t")
            setup_page.loc[setup_index, 'logdir'] = 'runs/sakt_legacy/' + dataset + '/'
            setup_page.loc[setup_index, 'savedir'] = 'save/sakt_legacy/' + dataset + '/'
            args = setup_page.loc[setup_index]
            stop_experiment = False # Stop current setup for whatever reason possible.
            if args[[f'test{i+1}' for i in range(REPEAT)]].notnull().all():
                continue
            for rand_seed in range(int(args['repeat'])):
                set_random_seeds(rand_seed)
                train_data, val_data = get_chunked_data(train_df, int(args.max_length), randomize=True, \
                    stride=int(args.stride), non_overlap_only=True)
                num_items = int(full_df["item_id"].max() + 1)
                num_skills = int(full_df["skill_id"].max() + 1)
                model = SAKT(num_items, num_skills, int(args.embed_size), int(args.num_attn_layers), int(args.num_heads),
                            bool(args.encode_pos), int(args.max_pos), args.drop_prob, query_feed=args.query_feed, \
                                query_highpass=args.query_highpass).cuda()
                if torch.cuda.device_count() > 1:
                    print('using {} GPUs'.format(torch.cuda.device_count()))
                    model = nn.DataParallel(model)
                model.to(torch.device("cuda"))
                param_str = '_'.join([str(x) + str(y) for x, y in args.to_dict().items()])[:200]
                logger = Logger(os.path.join(args.logdir, param_str), project_name='bt_sakt', \
                            run_name=dataset + ' ' + str(pd.datetime.now()), model_args=args.to_dict())
                while True: # Reduce batch size until it fits on GPU
                    try:    
                        optimizer = 'adam' if 'optimizer' not in args.index else args['optimizer']
                        saver = Saver(args.savedir, param_str, patience=10 if dataset in {'ednet', 'ednet_medium'} else 7)
                        train(train_data, val_data, model, optimizer, logger, saver, int(args.num_epochs),
                                int(args.batch_size), args.grad_clip)
                        logger.log_scalars({'adj_batch_size': args.batch_size}, step=logger.step)
                        break
                    except ValueError as e:
                        print(e)
                        args.loc['batch_size'] = args.batch_size // 2
                        setup_page.loc[setup_index, 'batch_size'] = args['batch_size']
                        print(f'Batch does not fit on gpu, reducing size to {args.batch_size}')
                        if args.batch_size < 25:
                            stop_experiment = True
                            break
                if stop_experiment:
                    break
                model = saver.load().cuda()
                
                if 1:
                    print('Testing...')
                    test_data, _ = get_chunked_data(test_df, int(args.max_length), train_split=1.0, \
                        randomize=False, stride=5, non_overlap_only=True)
                    test_batches = prepare_batches(test_data, batch_size=64, randomize=False)
                    test_preds = np.empty(0)
                    model.eval()
                    for item_inputs, skill_inputs, label_inputs, item_ids, skill_ids, labels in test_batches:
                        item_inputs = item_inputs.cuda()
                        skill_inputs = skill_inputs.cuda()
                        label_inputs = label_inputs.cuda()
                        item_ids = item_ids.cuda()
                        skill_ids = skill_ids.cuda()
                        with torch.no_grad():
                            preds = model(item_inputs, skill_inputs, label_inputs, item_ids, skill_ids)
                            preds = torch.sigmoid(preds[labels >= 0]).flatten().cpu().numpy()
                            test_preds = np.concatenate([test_preds, preds])
                    setup_score = roc_auc_score(test_df['correct'], test_preds)
                    print(setup_score)
                    logger.log_scalars({'auc/test': setup_score}, step=logger.step)
                    exp_ind = rand_seed + 1
                    setup_page.loc[setup_index, 'test{}'.format(exp_ind)] = setup_score
                    setup_page.loc[setup_index, 'valid{}'.format(exp_ind)] = saver.score_max
                    setup_page.loc[setup_index, 'best_epoch'] = saver.best_epoch

                    setup_page.to_excel(save_path.replace('setups/', 'results/'))
                    setup_page.to_excel(save_path)
                logger.close()
                del logger
                del model

        except ValueError as e:
            print(e)
            setup_page.loc[setup_index, 'remark'] = str(e)
            setup_page.to_excel(save_path.replace('setups/', 'results/'))

    setup_page.to_excel(save_path.replace('setups/', 'results/'))
    setup_page.to_excel(save_path)