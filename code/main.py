# Main file to be executed

import os
import sys
import json
import random
import argparse
import logging

from retry.api import retry_call
from tqdm import tqdm
from datetime import datetime
from painter import Trainer, NanException
from diff_augment_test import DiffAugmentTest

import torch
import torch.multiprocessing as mp
import torch.distributed as dist

import numpy as np


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def cast_list(el):
    return el if isinstance(el, list) else [el]


def timestamped_filename(prefix = 'generated-'):
    now = datetime.now()
    timestamp = now.strftime("%m-%d-%Y_%H-%M-%S")
    return f'{prefix}{timestamp}'


def set_seed(seed, loginf):
    loginf(f'Set seed: {seed}')
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def run_training(rank, world_size, model_args, data, load_from, new,
                 num_train_steps, name, seed, loginf, use_wandb,
                 load_only_g, reset_g_optimizer, reset_d_optimizer,
                 load_pre_trained=''):
    is_main = rank == 0
    is_ddp = world_size > 1
    set_seed(seed, loginf)

    if is_ddp:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        dist.init_process_group('nccl', rank=rank, world_size=world_size)

        print(f"{rank + 1}/{world_size} process initialized.")

    model_args.update(
        is_ddp = is_ddp,
        rank = rank,
        world_size = world_size
    )

    model = Trainer(**model_args, hparams=model_args)

    if not new:
        model.load(
            load_from, g_load_optimizer=not reset_g_optimizer,
            d_load_optimizer=not reset_d_optimizer, loginf=loginf)
    else:
        model.clear()

    # Load pre-trained FWP (to train with Unet)
    if load_pre_trained != '':
        model.clear()

        loginf(f'loading from {load_pre_trained}')

        pretrained_dict_all = torch.load(load_pre_trained)
        pretrained_dict = pretrained_dict_all['GAN']
        model_dict = model.GAN.state_dict()

        if not reset_g_optimizer:
            model.G_scaler.load_state_dict(pretrained_dict_all['G_scaler'])
            model.GAN.G_opt.load_state_dict(pretrained_dict_all['G_opt'])

        if not reset_d_optimizer:
            model.D_scaler.load_state_dict(pretrained_dict_all['D_scaler'])
            model.GAN.D_opt.load_state_dict(pretrained_dict_all['D_opt'])

        pretrained_dict = {
            k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict) 
        model.GAN.load_state_dict(pretrained_dict, strict=False)

    if load_only_g:
        loginf('loaded only Generator')
        from painter import Discriminator
        model.GAN.D = Discriminator(
            image_size = model_args['image_size'],
            fmap_max = model_args['fmap_max'],
            transparent = model_args['transparent'],
            greyscale = model_args['greyscale'],
            attn_res_layers = model_args['attn_res_layers'],
            disc_output_size = model_args['disc_output_size']
        )

    model.set_data_src(data)

    loginf(model.GAN)

    progress_bar = tqdm(
        initial = model.steps, total = num_train_steps, mininterval=10.,
        desc=f'{name}<{data}>')
    while model.steps < num_train_steps:
        retry_call(model.train, tries=3, exceptions=NanException)
        progress_bar.n = model.steps
        progress_bar.refresh()
        if is_main and model.steps % 50 == 0:
            model.print_log(loginf, use_wandb)

    model.save(model.checkpoint_num)

    if is_ddp:
        dist.destroy_process_group()


def train_from_folder(
    data = './data',
    image_save_format = 'jpg',
    results_dir = './results',
    models_dir = './models',
    name = 'default',
    new = False,
    save_best_and_last = True,
    load_from = -1,
    load_only_g = False,
    load_pre_trained = None,
    reset_g_optimizer = False,
    reset_d_optimizer = False,
    latent_dim = 256,
    image_size = 64,
    rnn_hidden_size = 512,
    num_layers = 1,
    num_decoding_steps = 64,
    unet_lr = 2e-4,
    num_heads = 8,
    dim_feedforward = 1024,
    mini_latent_dim = 8,
    mini_input_size = 128,
    wide_out_dim = 0,
    unet_tanh = False,
    dropout = 0.0,
    clip = 0.0,
    clip_rnn = 0.0,
    max_unet_dim = 4,
    out_tanh = False,
    use_pos_enc = False,
    use_softmax = False,
    no_sigmoid_lr = False,
    v2_input_sigmoid = False,
    val_tanh = False,
    vect_lr = False,
    use_latent_to_init = False,
    use_learned_init = False,
    use_intermed_loss = 1,
    scale_intermed_loss = True,
    use_pre_unet_loss = False,
    use_unet_match_loss = False,
    use_unet_match_loss_only = False,
    ema_only_unet = False,
    stop_grad_unet = False,
    frozen_base = False,
    rnn_init_fact = 1,
    kvb_init_fact = 1,
    model_type = 'base',
    optimizer = 'adam',
    fmap_max = 512,
    transparent = False,
    greyscale = False,
    skip_resize = False,
    transform_to_greyscale = False,
    batch_size = 10,
    gradient_accumulate_every = 4,
    num_train_steps = 150000,
    learning_rate = 2e-4,
    ttur_mult = 1.0,
    save_every = 1000,
    evaluate_every = 1000,
    generate = False,
    generate_types = ['default', 'ema'],
    generate_interpolation = False,
    aug_test = False,
    aug_prob=None,
    aug_types=['cutout', 'translation'],
    dataset_aug_prob=0.,
    attn_res_layers = [32],
    freq_chan_attn = False,
    disc_output_size = 1,
    dual_contrast_loss = False,
    antialias = False,
    interpolation_num_steps = 100,
    save_frames = False,
    num_image_tiles = None,
    num_workers = None,
    multi_gpus = False,
    calculate_fid_every = None,
    calculate_fid_num_images = 10000,
    eval_without_ema = False,
    fid_all_real = True,
    fid_real_path = None,
    clear_fid_cache = False,
    eval_fid = False,
    seed = 42,
    amp = False,
    show_progress = False,
    load_strict = True,
    loginf=None,
    use_wandb=False
):
    num_image_tiles = default(num_image_tiles, 4 if image_size > 512 else 8)

    assert loginf is not None

    model_args = dict(
        name = name,
        image_save_format = image_save_format,
        results_dir = results_dir,
        models_dir = models_dir,
        batch_size = batch_size,
        save_best_and_last = save_best_and_last,
        gradient_accumulate_every = gradient_accumulate_every,
        attn_res_layers = cast_list(attn_res_layers),
        freq_chan_attn = freq_chan_attn,
        disc_output_size = disc_output_size,
        dual_contrast_loss = dual_contrast_loss,
        antialias = antialias,
        latent_dim = latent_dim,
        image_size = image_size,
        rnn_hidden_size = rnn_hidden_size,
        num_layers = num_layers,
        mini_latent_dim = mini_latent_dim,
        mini_input_size = mini_input_size,
        wide_out_dim = wide_out_dim,
        num_heads = num_heads,
        dim_feedforward = dim_feedforward,
        dropout = dropout,
        clip = clip,
        clip_rnn = clip_rnn,
        max_unet_dim = max_unet_dim,
        frozen_base = frozen_base,
        rnn_init_fact = rnn_init_fact,
        kvb_init_fact = kvb_init_fact,
        unet_lr = unet_lr,
        unet_tanh = unet_tanh,
        num_decoding_steps = num_decoding_steps,
        out_tanh = out_tanh,
        use_pos_enc = use_pos_enc,
        use_softmax = use_softmax,
        no_sigmoid_lr = no_sigmoid_lr,
        v2_input_sigmoid = v2_input_sigmoid,
        val_tanh = val_tanh,
        vect_lr = vect_lr,
        use_latent_to_init = use_latent_to_init,
        use_learned_init = use_learned_init,
        use_intermed_loss = use_intermed_loss,
        scale_intermed_loss = scale_intermed_loss,
        use_pre_unet_loss = use_pre_unet_loss,
        use_unet_match_loss = use_unet_match_loss,
        use_unet_match_loss_only = use_unet_match_loss_only,
        stop_grad_unet = stop_grad_unet,
        model_type = model_type,
        ema_only_unet = ema_only_unet,
        num_image_tiles = num_image_tiles,
        optimizer = optimizer,
        num_workers = num_workers,
        fmap_max = fmap_max,
        transparent = transparent,
        greyscale = greyscale,
        skip_resize = skip_resize,
        transform_to_greyscale = transform_to_greyscale,
        lr = learning_rate,
        ttur_mult = ttur_mult,
        save_every = save_every,
        evaluate_every = evaluate_every,
        aug_prob = aug_prob,
        aug_types = cast_list(aug_types),
        dataset_aug_prob = dataset_aug_prob,
        calculate_fid_every = calculate_fid_every,
        calculate_fid_num_images = calculate_fid_num_images,
        fid_all_real = fid_all_real,
        fid_real_path = fid_real_path,
        clear_fid_cache = clear_fid_cache,
        amp = amp,
        load_strict = load_strict,
        use_wandb = use_wandb,
        loginf = loginf,
    )

    if eval_fid:
        model_args['calculate_fid_every'] = 1
        model = Trainer(**model_args)
        model.load(load_from, loginf=loginf)
        samples_name = timestamped_filename()
        checkpoint = model.checkpoint_num
        model.set_data_src(data)

        import math
        num_batches = math.ceil(
            model.calculate_fid_num_images / model.batch_size)
        fid = model.calculate_fid(
            num_batches, eval_without_ema, report_time=True)
        print(f'FID: {fid}')
        return

    if generate:
        model = Trainer(**model_args)
        model.load(load_from, loginf=loginf)

        samples_name = timestamped_filename()
        checkpoint = model.checkpoint_num
        if model_type == 'lstm_delta_unet':
            model.GAN.G.generate_steps = True
            model.GAN.GE.generate_steps = True
            dir_result = model.generate_steps(
                samples_name, num_image_tiles, checkpoint, generate_types,
                num_rows=num_decoding_steps+1)
        elif model_type == 'lstm_delta':
            model.GAN.G.generate_steps = True
            model.GAN.GE.generate_steps = True
            dir_result = model.generate_steps(
                samples_name, num_image_tiles, checkpoint, generate_types)
        elif model_type == 'lstm_delta_split_v2':
            model.GAN.G.generate_steps = True
            model.GAN.GE.generate_steps = True
            dir_result = model.generate_steps(
                samples_name, num_image_tiles, checkpoint, generate_types)
        elif model_type == 'lstm_delta_split_v2_unet':
            model.GAN.G.generate_steps = True
            model.GAN.GE.generate_steps = True
            dir_result = model.generate_steps(
                samples_name, num_image_tiles, checkpoint, generate_types,
                num_rows=num_decoding_steps+1)
        elif model_type == 'lstm_sum':
            model.GAN.G.generate_steps = True
            model.GAN.GE.generate_steps = True
            dir_result = model.generate_steps(
                samples_name, num_image_tiles, checkpoint, generate_types)
        else:
            dir_result = model.generate(
                samples_name, num_image_tiles, checkpoint, generate_types)
        print(f'sample images generated at {dir_result}')
        return

    if generate_interpolation:
        model = Trainer(**model_args)
        model.load(load_from, loginf=loginf)
        samples_name = timestamped_filename()
        model.generate_interpolation(
            samples_name, num_image_tiles, num_steps = interpolation_num_steps,
            save_frames = save_frames)
        print(f'interpolation generated: {results_dir}/{name}/{samples_name}')
        return

    if show_progress:
        model = Trainer(**model_args)
        model.show_progress(num_images=num_image_tiles, types=generate_types)
        return

    if aug_test:
        DiffAugmentTest(
            data=data, image_size=image_size, batch_size=batch_size,
            types=aug_types, nrow=num_image_tiles)
        return

    world_size = torch.cuda.device_count()

    if world_size == 1 or not multi_gpus:
        run_training(0, 1, model_args, data, load_from, new, num_train_steps,
                     name, seed, loginf, use_wandb, load_only_g,
                     reset_g_optimizer, reset_d_optimizer, load_pre_trained)
        return

    mp.spawn(run_training,
        args=(world_size, model_args, data, load_from, new, num_train_steps,
              name, seed, loginf, use_wandb,
              load_only_g, reset_g_optimizer, reset_d_optimizer,
              load_pre_trained),
        nprocs=world_size,
        join=True)


###############################################################################


parser = argparse.ArgumentParser(
    description='Fast Weight Painters in the lightweight GAN framework.')

parser.add_argument('--data_dir', type=str,
                    default='data/celeba/img_align_celeba',
                    help='location of the data corpus')
parser.add_argument('--result_dir', default='./results', type=str,
                    help='where to save model ckpt.')
parser.add_argument('--model_dir', default='./models', type=str,
                    help='where to save model ckpt.')
parser.add_argument('--exp_name', type=str, default='default')
parser.add_argument('--new', action='store_true')
parser.add_argument('--save_all', action='store_true',
                    help='save checkpoints each steps')

parser.add_argument('--transparent', action='store_true')
parser.add_argument('--greyscale', action='store_true')
parser.add_argument('--skip_resize', action='store_true',
                    help='skip image resize')
parser.add_argument('--transform_to_greyscale', action='store_true')

parser.add_argument('--latent_dim', default=256, type=int)
parser.add_argument('--image_size', default=64, type=int)
parser.add_argument('--rnn_hidden_size', default=512, type=int)
parser.add_argument('--num_layers', default=1, type=int)
parser.add_argument('--num_heads', default=8, type=int)
parser.add_argument('--ff_dim', default=1024, type=int)
parser.add_argument('--dropout', default=0.0, type=float)
parser.add_argument('--wide_out_dim', default=0, type=int)

# apply tanh before unet
parser.add_argument('--unet_tanh', action='store_true')

parser.add_argument('--use_intermed_loss', default=1, type=int,
                    help='use this many positions to compute loss')
parser.add_argument('--scale_intermed_loss', action='store_true')

parser.add_argument('--image_save_format', default='jpg', type=str,
                    choices=['jpg', 'png'])

parser.add_argument('--mini_latent_dim', default=8, type=int)
parser.add_argument('--mini_input_size', default=128, type=int)

parser.add_argument('--aug_prob', default=0.0, type=float)
parser.add_argument('--dataset_aug_prob', default=0.0, type=float)

parser.add_argument('--num_decoding_steps', default=64, type=int)
parser.add_argument('--out_tanh', action='store_true')
parser.add_argument('--use_learned_init', action='store_true')
parser.add_argument('--pre_unet_loss', action='store_true')
parser.add_argument('--stop_grad_unet', action='store_true')
parser.add_argument('--use_unet_match_loss', action='store_true')
parser.add_argument('--use_unet_match_loss_only', action='store_true')

parser.add_argument('--use_pos_enc', action='store_true')
parser.add_argument('--use_softmax', action='store_true')  # in the delta rule
parser.add_argument('--no_sigmoid_lr', action='store_true',
                    help='For delta models. Remove sigmoid on learning rate')
# this actually applies tanh, not sigmoid.
parser.add_argument('--v2_input_sigmoid', action='store_true',
                    help='only relevant for split and split_v2')
parser.add_argument('--val_tanh', action='store_true')
parser.add_argument('--vect_lr', action='store_true')
parser.add_argument('--use_latent_to_init', action='store_true')

# There are more models than those reported in the paper.
# The essential ones are:
# base: LightGAN baseline
# lstm_delta: FPA with v1 input generator
# lstm_delta_unet: U-Net extension of the model above
# lstm_delta_split_v2: FPA with v2 input generator
# lstm_delta_split_v2_unet: U-Net extension of the model above
parser.add_argument('--model_type', default='base', type=str,
                    choices=['base', 'lstm_delta', 'lstm_delta_unet',
                             'lstm_delta_split_v2', 'lstm_delta_split_v2_unet',
                             'lstm_delta_split', 'lstm_delta_split_unet',
                             'trafo_delta_split', 'rfwp_delta', 'lstm_sum',
                             'lstm_oja_split_v2'])
parser.add_argument('--seed', default=1, type=int, help='Seed.')
parser.add_argument('--optimizer', type=str, default='adam',
                    choices=['adam'])
parser.add_argument('--batch_size', default=10, type=int,
                    help='batch size.')
parser.add_argument('--grad_cummulate_every', default=4, type=int,
                    help='number of gradient accumulation steps.')
parser.add_argument('--load_from', default=-1, type=int,
                    help='ID of model to be loaded.')
parser.add_argument('--load_only_g', action='store_true')
parser.add_argument('--load_pre_trained', type=str, default='')

parser.add_argument('--reset_g_optimizer', action='store_true')
parser.add_argument('--reset_d_optimizer', action='store_true')

parser.add_argument('--ema_only_unet', action='store_true')

parser.add_argument('--num_image_tiles', default=8, type=int)

parser.add_argument('--frozen_base', action='store_true')

parser.add_argument('--generate', action='store_true')
parser.add_argument('--eval_fid', action='store_true')

parser.add_argument('--generate_interpolation', action='store_true')
parser.add_argument('--aug_test', action='store_true')

parser.add_argument('--generate_types', default=['default', 'ema'])
parser.add_argument('--multi_gpus', action='store_true')
parser.add_argument('--num_workers', default=None)

parser.add_argument('--eval_without_ema', action='store_true')

parser.add_argument('--rnn_init_fact', default=1, type=float,
                    help='scaling for init.')
parser.add_argument('--kvb_init_fact', default=1, type=float,
                    help='scaling for init.')
# training hyper-parameters:
parser.add_argument('--num_train_steps', default=150000, type=int,
                    help='Number of training steps to train on')
parser.add_argument('--learning_rate', default=2e-4, type=float,
                    help='learning rate.')
parser.add_argument('--ttur_mult', default=1, type=float,
                    help='discriminator lr = this * learning rate.')
parser.add_argument('--unet_lr', default=2e-4, type=float,
                    help='unet learning rate.')
parser.add_argument('--max_unet_dim', default=4, type=int,
                    help='unet dim max multiplier')
parser.add_argument('--clip', default=0.0, type=float,
                    help='global clipping rate.')
parser.add_argument('--clip_rnn', default=0.0, type=float,
                    help='RNN clipping rate.')
parser.add_argument('--save_every', default=1000, type=int,
                    help='Report log every this steps (not used).')
parser.add_argument('--evaluate_every', default=1000, type=int,
                    help='Report log every this steps (not used).')
parser.add_argument('--calculate_fid_every', default=None,
                    help='Report log every this steps (not used).')
parser.add_argument('--interpolation_num_steps', default=100, type=int)

# GAN
parser.add_argument('--fmap_max', default=512, type=int)

# eval params
parser.add_argument('--calculate_fid_num_images', default=10000, type=int)
parser.add_argument('--clear_fid_cache', action='store_true')
parser.add_argument('--not_fid_all_real', action='store_true')
parser.add_argument('--fid_real_path', type=str,
                    default='results/default/real',
                    help='location of the real resized images for FID')

# for wandb
parser.add_argument('--project_name', type=str, default=None,
                    help='project name for wandb.')
parser.add_argument('--job_name', type=str, default=None,
                    help='job name for wandb.')
parser.add_argument('--use_wandb', action='store_true',
                    help='use wandb.')

args = parser.parse_args()

args.work_dir = f"{args.result_dir}/{args.exp_name}"
if not os.path.exists(args.work_dir):
    os.makedirs(args.work_dir)

work_dir_key = '/'.join(os.path.abspath(args.work_dir).split('/')[-3:])

# logging
log_file_name = f"{args.work_dir}/log.txt"
handlers = [logging.FileHandler(log_file_name), logging.StreamHandler()]
logging.basicConfig(
    level=logging.INFO, format='%(message)s', handlers=handlers)

loginf = logging.info

loginf(f"torch version: {torch.__version__}")
loginf(f"Result dir: {args.work_dir}")

# wandb
if args.use_wandb:  # configure wandb.
    import wandb
    use_wandb = True

    if args.project_name is None:
        project_name = (os.uname()[1]
                        + datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    else:
        project_name = args.project_name

    wandb.init(
        project=project_name, settings=wandb.Settings(start_method='fork'))
    # or `settings=wandb.Settings(start_method='thread')`
    if args.job_name is None:
        wandb.run.name = f"{os.uname()[1]}//" \
                         f"{args.model_type}/{args.exp_name}//" \
                         f"seed{args.seed}//aug{args.aug_prob}/" \
                         f"lr{args.learning_rate}/ttur{args.ttur_mult}/" \
                         f"init{args.use_learned_init}/ucap{args.max_unet_dim}/" \
                         f"uloss{args.pre_unet_loss}/stopg{args.stop_grad_unet}/" \
                         f"ulr{args.unet_lr}/size{args.image_size}/" \
                         f"latent{args.latent_dim}/" \
                         f"steps{args.num_decoding_steps}//" \
                         f"rnn{args.rnn_hidden_size}/l{args.num_layers}/" \
                         f"H{args.num_heads}/ff{args.ff_dim}/" \
                         f"dr{args.dropout}/pos{args.use_pos_enc}/" \
                         f"smx{args.use_softmax}/siglr{args.no_sigmoid_lr}/" \
                         f"v2sig{args.v2_input_sigmoid}/vtan{args.val_tanh}/" \
                         f"vectlr{args.vect_lr}/clp{args.clip}/" \
                         f"clprnn{args.clip_rnn}/" \
                         f"latinit{args.use_latent_to_init}/B{args.batch_size}/" \
                         f"acc{args.grad_cummulate_every}" \
                         f"/tahn{args.out_tanh}/wide{args.wide_out_dim}/" \
                         f"scale{args.scale_intermed_loss}//" \
                         f"PATH'{work_dir_key}'//{os.environ.get('STY', 'none')}"
    else:
        wandb.run.name = f"{os.uname()[1]}//{args.job_name}"

    config = wandb.config
    config.host = os.uname()[1]  # host node name
    config.data_dir = args.data_dir
    config.result_dir = args.result_dir
    config.model_dir = args.model_dir
    config.exp_name = args.exp_name
    config.pre_unet_loss = args.pre_unet_loss
    config.stop_grad_unet = args.stop_grad_unet
    config.new = args.new
    config.save_all = args.save_all
    config.transparent = args.transparent
    config.skip_resize = args.skip_resize
    config.greyscale = args.greyscale
    config.transform_to_greyscale = args.transform_to_greyscale
    config.latent_dim = args.latent_dim
    config.image_size = args.image_size
    config.rnn_hidden_size = args.rnn_hidden_size
    config.num_layers = args.num_layers
    config.num_heads = args.num_heads
    config.ff_dim = args.ff_dim
    config.dropout = args.dropout
    config.clip = args.clip
    config.clip_rnn = args.clip_rnn
    config.unet_lr = args.unet_lr
    config.dataset_aug_prob = args.dataset_aug_prob
    config.aug_prob = args.aug_prob
    config.num_decoding_steps = args.num_decoding_steps
    config.out_tanh = args.out_tanh
    config.use_learned_init = args.use_learned_init
    config.use_pos_enc = args.use_pos_enc
    config.use_softmax = args.use_softmax
    config.no_sigmoid_lr = args.no_sigmoid_lr
    config.v2_input_sigmoid = args.v2_input_sigmoid
    config.val_tanh = args.val_tanh
    config.vect_lr = args.vect_lr
    config.use_latent_to_init = args.use_latent_to_init
    config.model_type = args.model_type
    config.seed = args.seed
    config.optimizer = args.optimizer
    config.batch_size = args.batch_size
    config.grad_cummulate_every = args.grad_cummulate_every
    config.load_from = args.load_from
    config.load_only_g = args.load_only_g
    config.reset_g_optimizer = args.reset_g_optimizer
    config.reset_d_optimizer = args.reset_d_optimizer
    config.generate = args.generate
    config.generate_interpolation = args.generate_interpolation
    config.aug_test = args.aug_test
    config.generate_types = args.generate_types
    config.multi_gpus = args.multi_gpus
    config.num_workers = args.num_workers
    config.num_train_steps = args.num_train_steps
    config.learning_rate = args.learning_rate
    config.ttur_mult = args.ttur_mult
    config.save_every = args.save_every
    config.evaluate_every = args.evaluate_every
    config.calculate_fid_every = args.calculate_fid_every
    config.interpolation_num_steps = args.interpolation_num_steps
    config.fmap_max = args.fmap_max
    config.calculate_fid_num_images = args.calculate_fid_num_images
    config.clear_fid_cache = args.clear_fid_cache
    config.max_unet_dim = args.max_unet_dim
    config.frozen_base = args.frozen_base
    config.rnn_init_fact = args.rnn_init_fact
    config.kvb_init_fact = args.kvb_init_fact
    config.image_save_format = args.image_save_format
    config.wide_out_dim = args.wide_out_dim
    config.use_intermed_loss = args.use_intermed_loss
    config.scale_intermed_loss = args.scale_intermed_loss
    config.unet_tanh = args.unet_tanh
    config.ema_only_unet = args.ema_only_unet
else:
    use_wandb = False
# end wandb

# save args
loginf(f"Command executed: {sys.argv[:]}")
loginf(f"Args: {json.dumps(args.__dict__, indent=2)}")

with open(f'{args.work_dir}/args.txt', 'w') as f:
    json.dump(args.__dict__, f, indent=2)

loginf(f"Seed: {args.seed}")


if args.calculate_fid_every is not None:
    calculate_fid_every = int(args.calculate_fid_every)
else:
    calculate_fid_every = None

if args.num_workers is not None:
    num_workers = args.num_workers
else:
    num_workers = None

save_best_and_last = not args.save_all

train_from_folder(
    data = args.data_dir,
    image_save_format = args.image_save_format,
    results_dir = args.result_dir,
    models_dir = args.model_dir,
    name = args.exp_name,
    new = args.new,
    save_best_and_last = save_best_and_last,
    latent_dim=args.latent_dim,
    image_size = args.image_size,
    rnn_hidden_size = args.rnn_hidden_size,
    num_layers = args.num_layers,
    num_heads = args.num_heads,
    dim_feedforward = args.ff_dim,
    mini_latent_dim = args.mini_latent_dim,
    mini_input_size = args.mini_input_size,
    wide_out_dim = args.wide_out_dim,
    unet_tanh = args.unet_tanh,
    dropout = args.dropout,
    clip = args.clip,
    clip_rnn = args.clip_rnn,
    max_unet_dim = args.max_unet_dim,
    unet_lr = args.unet_lr,
    num_decoding_steps = args.num_decoding_steps,
    out_tanh = args.out_tanh,
    use_pos_enc = args.use_pos_enc,
    use_softmax = args.use_softmax,
    no_sigmoid_lr = args.no_sigmoid_lr,
    v2_input_sigmoid = args.v2_input_sigmoid,
    val_tanh = args.val_tanh,
    vect_lr = args.vect_lr,
    use_latent_to_init = args.use_latent_to_init,
    use_learned_init = args.use_learned_init,
    use_intermed_loss = args.use_intermed_loss,
    scale_intermed_loss = args.scale_intermed_loss,
    use_pre_unet_loss = args.pre_unet_loss,
    use_unet_match_loss = args.use_unet_match_loss,
    use_unet_match_loss_only = args.use_unet_match_loss_only,
    ema_only_unet = args.ema_only_unet,
    stop_grad_unet = args.stop_grad_unet,
    frozen_base = args.frozen_base,
    rnn_init_fact = args.rnn_init_fact,
    kvb_init_fact = args.kvb_init_fact,
    model_type = args.model_type,
    transparent = args.transparent,
    skip_resize = args.skip_resize,
    greyscale = args.greyscale,
    transform_to_greyscale = args.transform_to_greyscale,
    optimizer = args.optimizer,
    batch_size = args.batch_size,
    gradient_accumulate_every = args.grad_cummulate_every,
    num_train_steps = args.num_train_steps,
    learning_rate = args.learning_rate,
    ttur_mult = args.ttur_mult,
    save_every = args.save_every,
    evaluate_every = args.evaluate_every,
    fmap_max = args.fmap_max,
    seed = args.seed,
    load_from = args.load_from,
    load_only_g = args.load_only_g,
    load_pre_trained = args.load_pre_trained,
    reset_g_optimizer = args.reset_g_optimizer,
    reset_d_optimizer = args.reset_d_optimizer,
    generate = args.generate,
    generate_types = args.generate_types,
    generate_interpolation = args.generate_interpolation,
    aug_test = args.aug_test,
    multi_gpus = args.multi_gpus,
    num_workers = num_workers,
    calculate_fid_num_images = args.calculate_fid_num_images,
    fid_all_real = not args.not_fid_all_real,
    fid_real_path = args.fid_real_path,
    calculate_fid_every = calculate_fid_every,
    eval_without_ema = args.eval_without_ema,
    eval_fid = args.eval_fid,
    interpolation_num_steps = args.interpolation_num_steps,
    clear_fid_cache = args.clear_fid_cache,
    aug_prob = args.aug_prob,
    aug_types=['cutout', 'translation'],
    dataset_aug_prob=args.dataset_aug_prob,
    attn_res_layers = [32],
    freq_chan_attn = False,
    disc_output_size = 1,
    dual_contrast_loss = False,
    antialias = False,
    save_frames = False,
    num_image_tiles = args.num_image_tiles,
    amp = False,
    show_progress = False,
    load_strict = True,
    loginf=loginf,
    use_wandb=args.use_wandb)
