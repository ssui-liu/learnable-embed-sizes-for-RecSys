import os
import numpy as np
import pandas as pd
from datetime import datetime
from argparse import ArgumentParser
from tensorboardX import SummaryWriter

from models.factorizer import setup_factorizer
from data_loader.data_loader import setup_generator
from utils.evaluate import evaluate_fm


def setup_args(parser=None):
    """ Set up arguments for the Engine

    return:
        python dictionary
    """
    if parser is None:
        parser = ArgumentParser()
    data = parser.add_argument_group('Data')
    engine = parser.add_argument_group('Engine Arguments')
    factorize = parser.add_argument_group('Factorizer Arguments')
    matrix_factorize = parser.add_argument_group('MF Arguments')
    regularize = parser.add_argument_group('Regularizer Arguments')
    log = parser.add_argument_group('Tensorboard Arguments')

    engine.add_argument('--alias', default='experiment',
                        help='Name for the experiment')
    engine.add_argument('--seed', default='42')

    data.add_argument('--data-type', default='ml1m', help='type of the dataset')
    data.add_argument('--data-path', default='./data/{data_type}/')
    data.add_argument('--train_test-freq-bd', help='split the data freq-wise, bound of the user freq')
    data.add_argument('--train-valid-freq-bd', help='split the data freq-wise, bound of the user freq')
    data.add_argument('--batch-size-train', default=1)
    data.add_argument('--batch-size-valid', default=1)
    data.add_argument('--batch-size-test', default=1)
    data.add_argument('--device-ids-test', default=[0], help='devices used for multi-processing evaluate')

    regularize.add_argument('--max-steps', default=1e8)
    regularize.add_argument('--use-cuda', default=True)
    regularize.add_argument('--device-id', default=0, help='Training Devices')

    factorize.add_argument('--factorizer', default='fm', help='Type of the Factorization Model')
    factorize.add_argument('--latent-dim', default=8)

    type_opt = 'fm'
    matrix_factorize.add_argument('--{}-optimizer'.format(type_opt), default='sgd')
    matrix_factorize.add_argument('--{}-lr'.format(type_opt), default=1e-3)
    matrix_factorize.add_argument('--{}-grad-clip'.format(type_opt), default=1)

    log.add_argument('--log-interval', default=1)
    log.add_argument('--tensorboard', default='./tmp/runs')
    log.add_argument('--early_stop', default=None)
    log.add_argument('--display_interval', default=100)
    return parser


class Engine(object):
    """Engine wrapping the training & evaluation
       of adpative regularized maxtirx factorization
    """

    def __init__(self, opt):
        self._opt = opt
        self._opt['data_path'] = self._opt['data_path'].format(data_type=self._opt['data_type'])
        self._sampler = setup_generator(opt)

        self._opt['field_dims'] = self._sampler.field_dims

        self._opt['emb_save_path'] = self._opt['emb_save_path'].format(
            factorizer=self._opt['factorizer'],
            data_type=self._opt['data_type'],
            alias=self._opt['alias'],
            num_parameter='{num_parameter}'
        )
        if 'retrain_emb_param' in opt:
            self.retrain = True
            if opt['re_init']:
                self._opt['alias'] += '_reinitTrue'
            else:
                self._opt['alias'] += '_reinitFalse'
            self._opt['alias'] += '_retrain_emb_param{}'.format(opt['retrain_emb_param'])
        else:
            self.retrain = False
            self.candidate_p = self._opt.get('candidate_p')
        self._opt['eval_res_path'] = self._opt['eval_res_path'].format(
            factorizer=self._opt['factorizer'],
            data_type=self._opt['data_type'],
            alias=self._opt['alias'],
            epoch_idx='{epoch_idx}'
        )
        self._factorizer = setup_factorizer(opt)
        self._opt['tensorboard'] = self._opt['tensorboard'].format(
            factorizer=self._opt['factorizer'],
            data_type=self._opt['data_type'],
        )
        self._writer = SummaryWriter(log_dir='{}/{}'.format(self._opt['tensorboard'], opt['alias']))
        self._writer.add_text('option', str(opt), 0)
        self._mode = None
        self.early_stop = self._opt.get('early_stop')


    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, new_mode):
        assert new_mode in ['complete', 'partial', None]  # training a complete trajectory or a partial trajctory
        self._mode = new_mode

    def save_pruned_embedding(self, param, step_idx):
        max_candidate_p = max(self.candidate_p)
        if max_candidate_p == 0:
            print("Minimal target parameters achieved, stop pruning.")
            exit(0)
        else:
            if param <= max_candidate_p:
                embedding = self._factorizer.model.get_embedding()
                emb_save_path = self._opt['emb_save_path'].format(num_parameter=param)
                emb_save_dir, _ = os.path.split(emb_save_path)
                if not os.path.exists(emb_save_dir):
                    os.makedirs(emb_save_dir)
                np.save(emb_save_path, embedding)
                max_idx = self.candidate_p.index(max(self.candidate_p))
                self.candidate_p[max_idx] = 0
                print("*" * 80)
                print("Reach the target parameter: {}, save embedding with size: {}".format(max_candidate_p, param))
                print("*" * 80)
            elif step_idx == 0:
                embedding = self._factorizer.model.get_embedding()
                emb_save_path = self._opt['emb_save_path'].format(num_parameter='initial_embedding')
                emb_save_dir, _ = os.path.split(emb_save_path)
                if not os.path.exists(emb_save_dir):
                    os.makedirs(emb_save_dir)
                np.save(emb_save_path, embedding)
                print("*" * 80)
                print("Save the initial embedding table")
                print("*" * 80)

    def train_an_episode(self, max_steps, episode_idx=''):
        """Train a feature_based recommendation model"""
        assert self.mode in ['partial', 'complete']

        print('-' * 80)
        print('[{} episode {} starts!]'.format(self.mode, episode_idx))
        print('Initializing ...')
        self._factorizer.init_episode()

        log_interval = self._opt.get('log_interval')
        eval_interval = self._opt.get('eval_interval')
        display_interval = self._opt.get('display_interval')

        status = dict()
        flag, test_flag, valid_flag = 0, 0, 0
        valid_mf_loss, train_mf_loss = np.inf, np.inf
        best_valid_result = {"AUC": [0, 0], "LogLoss": [np.inf, 0]}
        best_test_result = {"AUC": [0, 0], "LogLoss": [np.inf, 0]}
        epoch_start = datetime.now()
        for step_idx in range(int(max_steps)):
            # Prepare status for current step
            status['done'] = False
            status['sampler'] = self._sampler
            train_mf_loss = self._factorizer.update(self._sampler)
            status['train_mf_loss'] = train_mf_loss

            # Logging & Evaluate on the Evaluate Set
            if self.mode == 'complete' and step_idx % log_interval == 0:
                epoch_idx = int(step_idx / self._sampler.num_batches_train)
                sparsity, params = self._factorizer.model.calc_sparsity()
                if not self.retrain:
                    self.save_pruned_embedding(params, step_idx)
                self._writer.add_scalar('train/step_wise/mf_loss', train_mf_loss, step_idx)
                self._writer.add_scalar('train/step_wise/sparsity', sparsity, step_idx)

                if step_idx % display_interval == 0:
                    print('[Epoch {}|Step {}|Flag {}|Sparsity {:.4f}|Params {}]'.format(epoch_idx,
                                                                                        step_idx % self._sampler.num_batches_train,
                                                                                        flag, sparsity, params))

                if step_idx % self._sampler.num_batches_train == 0:
                    threshold = self._factorizer.model.get_threshold()

                    self._writer.add_histogram('threshold/epoch_wise/threshold', threshold, epoch_idx)
                    self._writer.add_scalar('train/epoch_wise/sparsity', sparsity, epoch_idx)
                    self._writer.add_scalar('train/epoch_wise/params', params, epoch_idx)

                if (step_idx % self._sampler.num_batches_train == 0) and (epoch_idx % eval_interval == 0) and self.retrain:
                    print('Evaluate on test ...')
                    start = datetime.now()
                    eval_res_path = self._opt['eval_res_path'].format(epoch_idx=epoch_idx)
                    eval_res_dir, _ = os.path.split(eval_res_path)
                    if not os.path.exists(eval_res_dir):
                        os.makedirs(eval_res_dir)

                    use_cuda = self._opt['use_cuda']
                    logloss, auc = evaluate_fm(self._factorizer, self._sampler, use_cuda)
                    self._writer.add_scalar('test/epoch_wise/metron_auc', auc, epoch_idx)
                    self._writer.add_scalar('test/epoch_wise/metron_logloss', logloss, epoch_idx)
                    if logloss < best_test_result['LogLoss'][0]:
                        best_test_result['LogLoss'][0] = logloss
                        best_test_result['LogLoss'][1] = epoch_idx
                    if auc > best_test_result['AUC'][0]:
                        best_test_result['AUC'][0] = auc
                        best_test_result['AUC'][1] = epoch_idx
                        test_flag = 0
                    else:
                        test_flag += 1
                    pd.Series(best_test_result).to_csv(eval_res_path)
                    print("*" * 80)
                    print("Test AUC: {:4f} | Logloss: {:4f}".format(auc, logloss))
                    end = datetime.now()
                    print('Evaluate Time {} minutes'.format((end - start).total_seconds() / 60))
                    epoch_end = datetime.now()
                    dur = (epoch_end - epoch_start).total_seconds() / 60
                    epoch_start = datetime.now()
                    print('[Epoch {:4d}] train MF loss: {:04.8f}, '
                          'valid loss: {:04.8f}, time {:04.8f} minutes'.format(epoch_idx,
                                                                               train_mf_loss,
                                                                               valid_mf_loss,
                                                                               dur))
                    print("*"*80)

            flag = test_flag
            if self.early_stop is not None and flag >= self.early_stop:
                print("Early stop training process")
                print("Best performance on test data: ", best_test_result)
                print("Best performance on valid data: ", best_valid_result)
                self._writer.add_text('best_valid_result', str(best_valid_result), 0)
                self._writer.add_text('best_test_result', str(best_test_result), 0)
                exit()

    def train(self):
        self.mode = 'complete'
        self.train_an_episode(self._opt['max_steps'])


if __name__ == '__main__':
    opt = setup_args()
    engine = Engine(opt)
    engine.train()
