from syne_tune import Reporter
import pickle
import torch
from fairseq.trainer import Trainer
from fairseq import checkpoint_utils, distributed_utils, options, progress_bar, tasks, utils
import yaml
import pdb
from fairseq.evolution import validate_all
from latency_predictor import LatencyPredictor
import numpy as np
report = Reporter()

def get_parent_parser():
    parser = options.get_training_parser()
    print("here")
    parser.add_argument('--feature-norm', type=float, nargs='+', help='normalizing factor for each feature')
    parser.add_argument('--lat-norm', type=float, help='normalizing factor for latency')
    parser.add_argument('--ckpt-path', type=str, help='path to load latency predictor weights')

    parser.add_argument('--latency-constraint', type=float, default=-1, help='latency constraint')
    parser.add_argument('--valid-cnt-max', type=int, default=1e9, help='max number of sentences to use in validation set')

    parser.add_argument('--write-config-path', type=str, help='path to write out the searched best SubTransformer')
    print("here")
    options.add_generation_args(parser)
    print("here")
    return parser

def get_predictor_details(metric):
    if metric == "cpu_raspberrypi":
        feature_norm = [640, 6, 2048, 6, 640, 6, 2048, 6, 6, 2]
        lat_norm = 5000
        ckpt_path = "/work/dlclarge1/sukthank-modnas/hardware-aware-transformers/latency_dataset/predictors/wmt14ende_cpu_raspberrypi.pt"
    elif metric == "cpu_xeon":
        feature_norm = [640, 6, 2048, 6, 640, 6, 2048, 6, 6, 2]
        lat_norm = 300
        ckpt_path = "/work/dlclarge1/sukthank-modnas/hardware-aware-transformers/latency_dataset/predictors/wmt14ende_cpu_xeon.pt"
    elif metric == "gpu_titanxp":
        feature_norm = [640, 6, 2048, 6, 640, 6, 2048, 6, 6, 2]
        lat_norm = 200
        ckpt_path = "/work/dlclarge1/sukthank-modnas/hardware-aware-transformers/latency_dataset/predictors/wmt14ende_gpu_titanxp.pt"

    return feature_norm, lat_norm, ckpt_path

def objective(args, arch, metric):
    utils.import_user_module(args)
    #min_val_loss = 14
    #max_val_loss = 17
    #max_lat = 10000
    #min_lat = 2000
    utils.handle_save_path(args)
    assert args.max_tokens is not None or args.max_sentences is not None, \
        'Must specify batch size either with --max-tokens or --max-sentences'
    if torch.cuda.is_available() and not args.cpu:
       torch.cuda.set_device(args.device_id)
    task = tasks.setup_task(args)
    for valid_sub_split in args.valid_subset.split(','):
        task.load_dataset(valid_sub_split, combine=False, epoch=0)

    # Build model and criterion
    model = task.build_model(args)
    criterion = task.build_criterion(args)
    print(model)    
    trainer = Trainer(args, task, model, criterion)
    args.train_subset = 'valid' # no need to train, so just set a small subset to save loading time
    args.restore_file = "/work/dlclarge1/sukthank-modnas/hardware-aware-transformers/downloaded_models/HAT_wmt14ende_super_space0.pt"
    extra_state, epoch_itr = checkpoint_utils.load_checkpoint(args, trainer)
    valid_loss = validate_all(args, trainer, task, epoch_itr, [arch])
    feature_norm, lat_norm, ckpt_path = get_predictor_details(metric)
    latency_predictor = LatencyPredictor(
            feature_norm=feature_norm,
            lat_norm=lat_norm,
            ckpt_path=ckpt_path
        )
    latency_predictor.load_ckpt()
    latency = latency_predictor.predict_lat(arch)
    latency_normalized = latency
    valid_loss_normalized = valid_loss[0]
    # asset not nan
    print(valid_loss_normalized, latency_normalized)
    assert not np.isnan(valid_loss_normalized)
    assert not np.isnan(latency_normalized)
    report(error=valid_loss_normalized, latency=latency_normalized)



if __name__ == "__main__":
    import logging
    import argparse
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    max_encoder_layers = 6
    max_decoder_layers = 6
    parser = argparse.ArgumentParser('HAT search script', parents=[get_parent_parser()],add_help=False)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--encoder-embed-choice_', type=int, default=512)
    parser.add_argument("--st_checkpoint_dir", type=str, default=".")
    parser.add_argument('--decoder-embed-choice_', type=int, default=640)
    parser.add_argument('--encoder-layer-num-choice_', type=int, default=6)
    parser.add_argument('--decoder-layer-num-choice_', type=int, default=2)
    parser.add_argument('--metric', type=str, default= "cpu_raspberrypi")
    for i in range(max_encoder_layers):
        parser.add_argument(f'--encoder-ffn-embed-choice_{i}', type=int, default=1024)
        parser.add_argument(f'--encoder-self-attention-heads-choice_{i}', type=int, default=4)
    for i in range(max_decoder_layers):
        parser.add_argument(f'--decoder-ffn-embed-choice_{i}', type=int, default=1024)
        parser.add_argument(f'--decoder-self-attention-heads-choice_{i}', type=int, default=4)
        parser.add_argument(f'--decoder-ende-attention-heads-choice_{i}', type=int, default=4)
        parser.add_argument(f'--decoder-arbitrary-ende-attn-choice_{i}', type=int, default=1)

    args = options.parse_args_and_arch(parser)
    if args.pdb:
        pdb.set_trace()

    # one GPU is fast enough to do the search
    args.distributed_world_size = 1
                  
    # if search on CPU, use fp32 as default
    if args.cpu:
        args.fp16 = False
    arch_dict = {}
    arch_dict['encoder'] = {}
    arch_dict['decoder'] = {}
    arch_dict['encoder']['encoder_embed_dim'] = args.encoder_embed_choice_
    arch_dict['decoder']['decoder_embed_dim'] = args.decoder_embed_choice_
    arch_dict['encoder']['encoder_layer_num'] = args.encoder_layer_num_choice_
    arch_dict['decoder']['decoder_layer_num'] = args.decoder_layer_num_choice_
    arch_dict['encoder'][f'encoder_ffn_embed_dim'] = []
    arch_dict['encoder'][f'encoder_self_attention_heads'] = []
    arch_dict['decoder'][f'decoder_ffn_embed_dim'] = []
    arch_dict['decoder'][f'decoder_self_attention_heads'] = []
    arch_dict['decoder'][f'decoder_ende_attention_heads'] = []
    arch_dict['decoder'][f'decoder_arbitrary_ende_attn'] = []
    arch_dict = {}
    arch_dict['encoder'] = {}
    arch_dict['decoder'] = {}
    arch_dict['encoder']['encoder_embed_dim'] = args.encoder_embed_choice_
    arch_dict['decoder']['decoder_embed_dim'] = args.decoder_embed_choice_
    arch_dict['encoder']['encoder_layer_num'] = args.encoder_layer_num_choice_
    arch_dict['decoder']['decoder_layer_num'] = args.decoder_layer_num_choice_
    arch_dict['encoder'][f'encoder_ffn_embed_dim'] = []
    arch_dict['encoder'][f'encoder_self_attention_heads'] = []
    arch_dict['decoder'][f'decoder_ffn_embed_dim'] = []
    arch_dict['decoder'][f'decoder_self_attention_heads'] = []
    arch_dict['decoder'][f'decoder_ende_attention_heads'] = []
    arch_dict['decoder'][f'decoder_arbitrary_ende_attn'] = []
    for i in range(max_decoder_layers):
        arch_dict['decoder'][f'decoder_ffn_embed_dim'].append(args.__dict__[f'decoder_ffn_embed_choice_{i}'])
        arch_dict['decoder'][f'decoder_self_attention_heads'].append(args.__dict__[f'decoder_self_attention_heads_choice_{i}'])
        arch_dict['decoder'][f'decoder_ende_attention_heads'].append(args.__dict__[f'decoder_ende_attention_heads_choice_{i}'])
        arch_dict['decoder'][f'decoder_arbitrary_ende_attn'].append(args.__dict__[f'decoder_arbitrary_ende_attn_choice_{i}'])
    for i in range(max_encoder_layers):
        arch_dict['encoder'][f'encoder_ffn_embed_dim'].append(args.__dict__[f'encoder_ffn_embed_choice_{i}'])
        arch_dict['encoder'][f'encoder_self_attention_heads'].append(args.__dict__[f'encoder_self_attention_heads_choice_{i}'])
    objective(args, arch_dict, args.metric)