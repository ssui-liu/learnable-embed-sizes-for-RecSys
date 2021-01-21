from engine import setup_args, Engine
import torch
import os
import random
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.backends.cudnn.enabled = True

if __name__ == '__main__':
    parser = setup_args()
    parser.set_defaults(
        alias='test',
        tensorboard='./tmp/runs/{factorizer}/{data_type}',
        ##########
        ## data ##
        ##########
        data_type='avazu',
        data_path='./data/{data_type}/',
        load_in_queue=False,
        category_only=False,
        rebuild_cache=False,
        eval_res_path='./tmp/res/{factorizer}/{data_type}/{alias}/{epoch_idx}.csv',
        emb_save_path='./tmp/embedding/{factorizer}/{data_type}/{alias}/{num_parameter}',
        ######################
        ## train/test split ##
        ######################
        train_test_split='lro',
        test_ratio=0.1,
        valid_ratio=1/9,
        ##########################
        ## Devices & Efficiency ##
        ##########################
        use_cuda=True,
        early_stop=5,
        log_interval=1,  # 816
        eval_interval=1,
        display_interval=2000,  # 10 epochs between 2 evaluations
        device_ids_test=[0],
        device_id=0,
        batch_size_train=1024,
        batch_size_valid=1024,
        batch_size_test=1024,
        ###########
        ## Model ##
        ###########
        factorizer='fm',
        model='fm',
        fm_lr=1e-3,
        # Deep
        mlp_dims=[400, 400, 400],
        # AutoInt
        has_residual=True,
        full_part=True,
        num_heads=2,
        num_layers=3,
        att_dropout=0,
        atten_embed_dim=64,
        # optimizer setting
        fm_optimizer='adam',
        fm_amsgrad=False,
        fm_eps=1e-8,
        fm_l2_regularization=1e-5,
        fm_betas=(0.9, 0.999),
        fm_grad_clip=100,  # 0.1
        fm_lr_exp_decay=1,
        l2_penalty=0,
        #########
        ## PEP ##
        #########
        latent_dim=24,
        threshold_type='feature_dim',
        g_type='sigmoid',
        gk=1,
        threshold_init=-150,
        candidate_p=[50000, 30000, 20000],
    )

    opt = parser.parse_args(args=[])
    opt = vars(opt)

    # rename alias
    # rename alias

    opt['alias'] = '{}_{}_BaseDim{}_bsz{}_lr_{}_optim_{}_thresholdType{}_thres_init{}_{}-{}_l2_penalty{}'.format(
        opt['model'].upper(),
        opt['alias'],
        opt['latent_dim'],
        opt['batch_size_train'],
        opt['fm_lr'],
        opt['fm_optimizer'],
        opt['threshold_type'].upper(),
        opt['threshold_init'],
        opt['g_type'],
        opt['gk'],
        opt['l2_penalty']
    )

    print(opt['alias'])
    random.seed(opt['seed'])
    # np.random.seed(opt['seed'])
    torch.manual_seed(opt['seed'])
    torch.cuda.manual_seed_all(opt['seed'])
    engine = Engine(opt)
    engine.train()
