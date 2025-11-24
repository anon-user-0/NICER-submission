import os
from os.path import join as j_
import pdb
import time
import torch.nn.functional as F

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

try:
    from sksurv.metrics import concordance_index_censored
except ImportError:
    print('scikit-survival not installed. Exiting...')
    raise

from sklearn.metrics import (roc_auc_score, balanced_accuracy_score,
                             cohen_kappa_score, classification_report, accuracy_score)

from mil_models.tokenizer import PrototypeTokenizer
from mil_models import create_downstream_model, prepare_emb
from utils.losses import NLLSurvLoss, CoxLoss, SurvRankingLoss
from utils.utils import (EarlyStopping, save_checkpoint, AverageMeter,
                         get_optim, print_network, get_lr_scheduler)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PROTO_MODELS = ['PANTHER', 'OT', 'H2T', 'ProtoCount', 'DeepSets',
                'NonParam_Local_Groups', 'InfiniteGPFA', 'KMeans', 'DPMeans', 'AdaptiveGMM']

## GENERIC
def log_dict_tensorboard(writer, results, str_prefix, step=0, verbose=False):
    for k, v in results.items():
        if verbose: print(f'{k}: {v:.4f}')
        writer.add_scalar(f'{str_prefix}{k}', v, step)
    return writer


def train(datasets, args, mode='classification'):
    """
    Train for a single fold for classification or suvival
    """
    
    writer_dir = args.results_dir
    if not os.path.isdir(writer_dir):
        os.mkdir(writer_dir)
    writer = SummaryWriter(writer_dir, flush_secs=15)

    assert args.es_metric == 'loss'
    if mode == 'classification':
        loss_fn = nn.CrossEntropyLoss()

    elif mode == 'survival':
        if args.loss_fn == 'nll':
            loss_fn = NLLSurvLoss(alpha=args.nll_alpha)
        elif args.loss_fn == 'cox':
            loss_fn = CoxLoss()
        elif args.loss_fn == 'rank':
            loss_fn = SurvRankingLoss()

    print('\nInit Model...', end=' ')

    # If prototype-based models, need to create slide-level embeddings
    if (args.model_type in PROTO_MODELS) or args.dd_test:
        datasets, _ = prepare_emb(datasets, args, mode)

        new_in_dim = None
        for k, loader in datasets.items():
            if (args.dd_test) and ('train' not in k) and ('val' not in k): continue   # only deal with the proposal
            
            assert loader.dataset.X is not None
            new_in_dim_curr = loader.dataset.X.shape[-1]
            if new_in_dim is None:
                new_in_dim = new_in_dim_curr
            else:
                assert new_in_dim == new_in_dim_curr
            if 'LinearEmb' in args.emb_model_type:
                # Theis emb_model_type doesn't require per-prototype structure (simple linear layer)
                factor = 1
            elif args.dd_test and 'PANTHER' not in args.model_type and 'OT' not in args.model_type:
                factor = 1
            else:
                if 'NonParam' in args.model_type:
                    factor=1
                    
                elif 'GPFA' in args.model_type:
                    factor=1
                else:
                    # The original embedding is 1-D (long) feature vector
                    # Reshape it to (n_proto, -1)
                    tokenizer = PrototypeTokenizer(args.model_type, args.out_type, args.n_proto)
                    prob, mean, cov = tokenizer(loader.dataset.X)
                    
                    if 'OT' in args.model_type:
                        tensorized_X = [torch.Tensor(mean)]
                    else:
                        tensorized_X = [torch.Tensor(prob).unsqueeze(dim=-1), torch.Tensor(mean), torch.Tensor(cov)]
                    
                    loader.dataset.X = torch.cat(tensorized_X, dim=-1)

                    factor = args.n_proto
            
        args.in_dim = new_in_dim // factor

        args.model_type = args.emb_model_type
        args.model_config = args.emb_model_type # actually the config
    else:
        print(f"{args.model_type} doesn't construct unsupervised slide-level embeddings!")

    model = create_downstream_model(args, mode=mode)
    model.to(device)
    print('Done!')
    print_network(model)

    print('\nInit optimizer ...', end=' ')
    optimizer = get_optim(model=model, args=args)
    lr_scheduler = get_lr_scheduler(args, optimizer, datasets['train'])

    if args.early_stopping:
        print('\nSetup EarlyStopping...', end=' ')
        early_stopper = EarlyStopping(save_dir=args.results_dir,
                                      patience=args.es_patience,
                                      min_stop_epoch=args.es_min_epochs,
                                      better='min' if args.es_metric == 'loss' else 'max',
                                      verbose=True)
    else:
        print('\nNo EarlyStopping...', end=' ')
        early_stopper = None
    
    #####################
    # The training loop #
    #####################
    
    best_f1 = 0.0
    best_ci = 0.0
    for epoch in range(args.max_epochs):
        step_log = {'epoch': epoch, 'samples_seen': (epoch + 1) * len(datasets['train'].dataset)}

        ### Train Loop
        print('#' * 10, f'TRAIN Epoch: {epoch}', '#' * 10)
        if mode == 'classification':
            train_results = train_loop_classification(model, datasets['train'], optimizer, lr_scheduler, loss_fn,
                                                      in_dropout=args.in_dropout, print_every=args.print_every,
                                                      accum_steps=args.accum_steps)
        elif mode == 'survival':
            train_results = train_loop_survival(model, datasets['train'], optimizer, lr_scheduler, loss_fn,
                                                in_dropout=args.in_dropout, print_every=args.print_every,
                                                accum_steps=args.accum_steps)

        writer = log_dict_tensorboard(writer, train_results, 'train/', epoch)

        ### Validation Loop (Optional)
        if 'val' in datasets.keys():
            print('#' * 11, f'VAL Epoch: {epoch}', '#' * 11)
            if mode == 'classification':
                val_results, _ = validate_classification(model, datasets['val'], loss_fn,
                                                         print_every=args.print_every, verbose=True)
                
                if val_results['weighted_f1'] > best_f1:
                    best_f1 = val_results['weighted_f1']
                    best_ckpt_kwargs = dict(config=vars(args),
                                            epoch=epoch,
                                            model=model,
                                            score=best_f1,
                                            save_dir=args.results_dir,
                                            fname=f'best_checkpoint.pth')
                    save_checkpoint(**best_ckpt_kwargs)
                    
                    print(f"[CKPT] Save CKPT with F1 {best_f1} to {args.results_dir}\n")
                
                
            elif mode == 'survival':
                val_results, _ = validate_survival(model, datasets['val'], loss_fn,
                                                   print_every=args.print_every, verbose=True)
                if val_results['c_index'] > best_ci:
                    best_ci = val_results['c_index']
                    best_ckpt_kwargs = dict(config=vars(args),
                                            epoch=epoch,
                                            model=model,
                                            score=best_ci,
                                            save_dir=args.results_dir,
                                            fname=f'best_checkpoint.pth')
                    save_checkpoint(**best_ckpt_kwargs)
                    
                    print(f"[CKPT] Save CKPT with C-index {best_ci}\n")

            writer = log_dict_tensorboard(writer, val_results, 'val/', epoch)

            ### Check Early Stopping (Optional)
            if early_stopper is not None:
                if args.es_metric == 'loss':
                    score = val_results['loss']

                else:
                    raise NotImplementedError
                save_ckpt_kwargs = dict(config=vars(args),
                                        epoch=epoch,
                                        model=model,
                                        score=score,
                                        fname=f's_checkpoint.pth')
                stop = early_stopper(epoch, score, save_checkpoint, save_ckpt_kwargs)
                if stop:
                    break
        print('#' * (22 + len(f'TRAIN Epoch: {epoch}')), '\n')

    ### End of epoch: Load in the best model (or save the latest model with not early stopping)
    if args.early_stopping:
        model.load_state_dict(torch.load(j_(args.results_dir, f"s_checkpoint.pth"))['model'])
    else:
        torch.save(model.state_dict(), j_(args.results_dir, f"s_checkpoint.pth"))
        model.load_state_dict(torch.load(j_(args.results_dir, f"best_checkpoint.pth"),
                                         weights_only=False)['model'])

    ### End of epoch: Evaluate on val and test set
    results, dumps = {}, {}
    for k, loader in datasets.items():
        print(f'End of training. Evaluating on Split {k.upper()}...:')
        if mode == 'classification':
            results[k], dumps[k] = validate_classification(model, loader, loss_fn, print_every=args.print_every,
                                                           dump_results=True, verbose=False)
        elif mode == 'survival':
            results[k], dumps[k] = validate_survival(model, loader, loss_fn, print_every=args.print_every,
                                                     dump_results=True, verbose=False)

        if k == 'train':
            _ = results.pop('train')  # Train results by default are not saved in the summary, but train dumps are
        else:
            log_dict_tensorboard(writer, results[k], f'final/{k}_', 0, verbose=True)

    writer.close()
    return results, dumps


## CLASSIFICATION
def train_loop_classification(model, loader, optimizer, lr_scheduler, loss_fn=None,
                              in_dropout=0.0, print_every=50,
                              accum_steps=1):
    model.train()
    meters = {'bag_size': AverageMeter(), 'cls_acc': AverageMeter()}
    bag_size_meter = meters['bag_size']
    acc_meter = meters['cls_acc']

    #import pdb; pdb.set_trace()
    mean_time = 0
    for batch_idx, batch in enumerate(loader):
        start_time = time.time()
        data = batch['img'].to(device)
        label = batch['label'].to(torch.long).to(device)
        if len(label.shape) == 2 and label.shape[1] == 1:
            label = label.squeeze(dim=-1)

        if in_dropout:
            data = F.dropout(data, p=in_dropout)
        attn_mask = batch['attn_mask'].to(device) if ('attn_mask' in batch) else None
        model_kwargs = {'attn_mask': attn_mask, 'label': label, 'loss_fn': loss_fn}
        out, log_dict = model(data, model_kwargs)
        
        

        # Get loss + backprop
        loss = out['loss']
        loss = loss / accum_steps
        loss.backward()
        if (batch_idx + 1) % accum_steps == 0:
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
        end_time = time.time()
        mean_time += (end_time - start_time)

        # End of iteration classification-specific metrics to calculate / log
        logits = out['logits']
        acc = (label == logits.argmax(dim=-1)).float().mean()

        for key, val in log_dict.items():
            if key not in meters:
                meters[key] = AverageMeter()
            meters[key].update(val, n=len(data))

        acc_meter.update(acc.item(), n=len(data))
        bag_size_meter.update(data.size(1), n=len(data))

        if ((batch_idx + 1) % print_every == 0) or (batch_idx == len(loader) - 1):
            msg = [f"avg_{k}: {meter.avg:.4f}" for k, meter in meters.items()]
            msg = f"batch {batch_idx}\t" + "\t".join(msg)
            print(msg)

    print(f"Average forward time per batch: {mean_time / (batch_idx + 1)} seconds")
    # End of epoch classification-specific metrics to calculate / log
    results = {k: meter.avg for k, meter in meters.items()}
    results['lr'] = optimizer.param_groups[0]['lr']
    return results

@torch.no_grad()
def validate_classification(model, loader,
                            loss_fn=None,
                            print_every=50,
                            dump_results=False,
                            verbose=1):
    model.eval()
    meters = {'bag_size': AverageMeter(), 'cls_acc': AverageMeter()}
    acc_meter = meters['cls_acc']
    bag_size_meter = meters['bag_size']
    all_probs = []
    all_labels = []
    all_filenames = []
    
    mean_time = 0.0
    for batch_idx, batch in enumerate(loader):
        all_filenames.extend(batch['filename'][0])
        data = batch['img'].to(device)
        print(data.shape)
        label = batch['label'].to(torch.long).to(device)
        if len(label.shape) == 2 and label.shape[1] == 1:
            label = label.squeeze(dim=-1)

        start_time = time.time()
        attn_mask = batch['attn_mask'].to(device) if ('attn_mask' in batch) else None
        model_kwargs = {'attn_mask': attn_mask, 'label': label, 'loss_fn': loss_fn}
        out, log_dict = model(data, model_kwargs)
        end_time = time.time()
        mean_time += (end_time - start_time)
        # End of iteration classification-specific metrics to calculate / log
        logits = out['logits']
        acc = (label == logits.argmax(dim=-1)).float().mean()
        acc_meter.update(acc.item(), n=len(data))
        bag_size_meter.update(data.size(1), n=len(data))
        for key, val in log_dict.items():
            if key not in meters:
                meters[key] = AverageMeter()
            meters[key].update(val, n=len(data))
        all_probs.append(torch.softmax(logits, dim=-1).cpu().numpy())
        all_labels.append(label.cpu().numpy())

        if verbose and (((batch_idx + 1) % print_every == 0) or (batch_idx == len(loader) - 1)):
            msg = [f"avg_{k}: {meter.avg:.4f}" for k, meter in meters.items()]
            msg = f"batch {batch_idx}\t" + "\t".join(msg)
            print(msg)
    print(f"Average eval time per batch: {mean_time / (batch_idx + 1)} seconds")

    # End of epoch classification-specific metrics to calculate / log
    n_classes = logits.size(1)
    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    all_preds = all_probs.argmax(axis=1)
    
    result_analysis_dict = {
        'FILENAME': all_filenames,
        'PREDICTION': all_preds,
        'LABEL': all_labels
    }
    result_df = pd.DataFrame(result_analysis_dict)
    # result_df.to_csv("wrong_instances_analysis.csv", index=False)

    results = sweep_classification_metrics(all_probs, all_labels, all_preds=all_preds, n_classes=n_classes)
    results.update({k: meter.avg for k, meter in meters.items()})

    if 'report' in results:
        del results['report']

    if verbose:
        msg = [f"{k}: {v:.3f}" for k, v in results.items()]
        print("\t".join(msg))

    dumps = {}
    if dump_results:
        dumps['labels'] = all_labels
        dumps['probs'] = all_probs
        dumps['sample_ids'] = np.array(
            loader.dataset.idx2sample_df['sample_id'])
    return results, dumps


## SURVIVAL
def train_loop_survival(model, loader, optimizer, lr_scheduler, loss_fn=None, in_dropout=0.0, print_every=50,
                        accum_steps=32):
    model.train()
    meters = {'bag_size': AverageMeter()}
    bag_size_meter = meters['bag_size']
    all_risk_scores, all_censorships, all_event_times = [], [], []
    
    for batch_idx, batch in enumerate(loader):
        data = batch['img'].to(device)
        label = batch['label'].to(device)

        if in_dropout:
            data = F.dropout(data, p=in_dropout)
        event_time = batch['survival_time'].to(device)
        censorship = batch['censorship'].to(device)
        attn_mask = batch['attn_mask'].to(device) if ('attn_mask' in batch) else None
        model_kwargs = {'attn_mask': attn_mask, 'label': label, 'censorship': censorship, 'loss_fn': loss_fn}
        out, log_dict = model(data, model_kwargs)

        # print(data.shape, attn_mask.shape if attn_mask is not None else 0)

        if out['loss'] is None:
            continue

        # Get loss + backprop
        loss = out['loss']
        loss = loss / accum_steps
        loss.backward()
        if (batch_idx + 1) % accum_steps == 0:
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        # End of iteration survival-specific metrics to calculate / log
        all_risk_scores.append(out['risk'].detach().cpu().numpy())
        all_censorships.append(censorship.cpu().numpy())
        all_event_times.append(event_time.cpu().numpy())

        for key, val in log_dict.items():
            if key not in meters:
                meters[key] = AverageMeter()
            meters[key].update(val, n=len(data))

        bag_size_meter.update(data.size(1), n=len(data))

        if ((batch_idx + 1) % print_every == 0) or (batch_idx == len(loader) - 1):
            msg = [f"avg_{k}: {meter.avg:.4f}" for k, meter in meters.items()]
            msg = f"batch {batch_idx}\t" + "\t".join(msg)
            print(msg)

    # End of epoch survival-specific metrics to calculate / log
    all_risk_scores = np.concatenate(all_risk_scores).squeeze(-1)
    all_censorships = np.concatenate(all_censorships).squeeze(1)
    all_event_times = np.concatenate(all_event_times).squeeze(1)
    c_index = concordance_index_censored(
        (1 - all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
    results = {k: meter.avg for k, meter in meters.items()}
    results.update({'c_index': c_index})
    results['lr'] = optimizer.param_groups[0]['lr']
    return results


@torch.no_grad()
def validate_survival(model, loader,
                      loss_fn=None,
                      print_every=50,
                      dump_results=False,
                      recompute_loss_at_end=True,
                      verbose=1):
    model.eval()
    meters = {'bag_size': AverageMeter()}
    bag_size_meter = meters['bag_size']
    all_risk_scores, all_censorships, all_event_times = [], [], []


    for batch_idx, batch in enumerate(loader):
        data = batch['img'].to(device)
        label = batch['label'].to(device)

        event_time = batch['survival_time'].to(device)
        censorship = batch['censorship'].to(device)
        attn_mask = batch['attn_mask'].to(device) if ('attn_mask' in batch) else None
        model_kwargs = {'attn_mask': attn_mask, 'label': label, 'censorship': censorship, 'loss_fn': loss_fn}
        out, log_dict = model(data, model_kwargs)

        # End of iteration survival-specific metrics to calculate / log
        bag_size_meter.update(data.size(1), n=len(data))
        for key, val in log_dict.items():
            if key not in meters:
                meters[key] = AverageMeter()
            meters[key].update(val, n=len(data))
        all_risk_scores.append(out['risk'].cpu().numpy())
        all_censorships.append(censorship.cpu().numpy())
        all_event_times.append(event_time.cpu().numpy())

        if verbose and (((batch_idx + 1) % print_every == 0) or (batch_idx == len(loader) - 1)):
            msg = [f"avg_{k}: {meter.avg:.4f}" for k, meter in meters.items()]
            msg = f"batch {batch_idx}\t" + "\t".join(msg)
            print(msg)

    # End of epoch survival-specific metrics to calculate / log
    all_risk_scores = np.concatenate(all_risk_scores).squeeze(1)
    all_censorships = np.concatenate(all_censorships).squeeze(1)
    all_event_times = np.concatenate(all_event_times).squeeze(1)

    c_index = concordance_index_censored(
        (1 - all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
    results = {k: meter.avg for k, meter in meters.items()}
    results.update({'c_index': c_index})

    if recompute_loss_at_end and isinstance(loss_fn, CoxLoss):
        surv_loss_dict = loss_fn(logits=torch.tensor(all_risk_scores).unsqueeze(1),
                                 times=torch.tensor(all_event_times).unsqueeze(1),
                                 censorships=torch.tensor(all_censorships).unsqueeze(1))
        results['surv_loss'] = surv_loss_dict['loss'].item()
        results.update({k: v.item() for k, v in surv_loss_dict.items() if isinstance(v, torch.Tensor)})

    if verbose:
        msg = [f"{k}: {v:.3f}" for k, v in results.items()]
        print("\t".join(msg))

    dumps = {}
    if dump_results:
        dumps['all_risk_scores'] = all_risk_scores
        dumps['all_censorships'] = all_censorships
        dumps['all_event_times'] = all_event_times
        dumps['sample_ids'] = np.array(
            loader.dataset.idx2sample_df['sample_id'])
    return results, dumps


@torch.no_grad()
def sweep_classification_metrics(all_probs, all_labels, all_preds=None, n_classes=None):
    if n_classes is None:
        n_classes = all_probs.shape[1]

    if all_preds is None:
        all_preds = all_probs.argmax(axis=1)

    if n_classes == 2:
        all_probs = all_probs[:, 1]
        roc_kwargs = {}
    else:
        roc_kwargs = {'multi_class': 'ovo', 'average': 'macro'}

    bacc = balanced_accuracy_score(all_labels, all_preds)
    kappa = cohen_kappa_score(all_labels, all_preds, weights='quadratic')
    cls_rep = classification_report(all_labels, all_preds, output_dict=True, zero_division=0)
    acc = accuracy_score(all_labels, all_preds)
    roc_auc = roc_auc_score(all_labels, all_probs, **roc_kwargs)

    results = {'acc': acc,
               'bacc': bacc,
               'report': cls_rep,
               'kappa': kappa,
               'roc_auc': roc_auc,
               'weighted_f1': cls_rep['weighted avg']['f1-score']}
    return results