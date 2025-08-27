import os 
import sys 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import argparse

from lbm.models.lbm_online import LBM_export
from lbm.utils.train_utils import fix_random_seeds
from lbm.utils.eval_utils import load_config


def main(args):
    """main
    """
    fix_random_seeds(42)
    config = load_config(args)

    # init tracking model
    model = LBM_export(config).cuda().eval()
    checkpoint = torch.load(config.checkpoint_path)
    model.load_state_dict(checkpoint['model'], strict=True)

    # prepare input data
    N = 10
    memory_size = 12
    frame = torch.rand(1, 3, config.input_size[0], config.input_size[1]).cuda()
    queries = torch.rand(1, N, config.embed_dim).cuda()
    collision_dist = torch.rand(1, N, memory_size, config.embed_dim).cuda()
    stream_dist = torch.rand(1, N, memory_size, config.embed_dim).cuda()
    vis_mask = torch.ones(1, N, memory_size, dtype=torch.bool).cuda()
    mem_mask = torch.ones(1, N, memory_size, dtype=torch.bool).cuda()
    last_pos = torch.rand(N, 2).cuda()
    _ = model(frame, queries, collision_dist, stream_dist, vis_mask, mem_mask, last_pos)

    dynamic_axes = {
        'queries': {1: 'N'},
        'collision_dist': {1: 'N', 2: 'M'},
        'stream_dist': {1: 'N', 2: 'M'},
        'vis_mask': {1: 'N', 2: 'M'},
        'mem_mask': {1: 'N', 2: 'M'},
        'last_pos': {0: 'N'},
    }

    torch.onnx.export(
        model, 
        (frame, queries, collision_dist, stream_dist, vis_mask, mem_mask, last_pos), 
        args.output_file,
        input_names=['frame', 'queries', 'collision_dist', 'stream_dist', 'vis_mask', 'mem_mask', 'last_pos'],
        output_names=['f_t', 'coord_pred', 'vis_pred', 'collision_dist_new', 'stream_dist_new', 'vis_mask_new', 'mem_mask_new', 'last_pos_new'],
        dynamic_axes=dynamic_axes,
        opset_version=16, 
        verbose=False,
        do_constant_folding=False,
    )

    if args.check:
        import onnx
        onnx_model = onnx.load(args.output_file)
        onnx.checker.check_model(onnx_model)
        print('Check export onnx model done...')

    if args.simplify:
        import onnx 
        import onnxsim
        dynamic = True 
        input_shapes = {'frame': frame.shape, 'queries': queries.shape, 'collision_dist': collision_dist.shape,
            'stream_dist': stream_dist.shape, 'vis_mask': vis_mask.shape, 'mem_mask': mem_mask.shape, 
            'last_pos': last_pos.shape} if dynamic else None
        onnx_model_simplify, check = onnxsim.simplify(args.output_file, input_shapes=input_shapes, dynamic_input_shape=dynamic)
        onnx.save(onnx_model_simplify, args.output_file)
        print(f'Simplify onnx model {check}...')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export LBM model to ONNX format.')
    parser.add_argument('--config_path', type=str, default='lbm/configs/demo.yaml', help='Path to the configuration file.')
    parser.add_argument('--checkpoint_path', type=str, default='checkpoints/lbm.pt', help='Path to the checkpoint file.')
    parser.add_argument('--output_file', '-o', type=str, default='checkpoints/lbm.onnx', help='Output ONNX file path')
    parser.add_argument('--check', action='store_true', default=True, help='Check ONNX model after export')
    parser.add_argument('--simplify', action='store_true', default=False, help='Simplify ONNX model after export')

    args = parser.parse_args()
    main(args)