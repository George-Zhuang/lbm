import torch
import os
import argparse

def get_args():
    parser = argparse.ArgumentParser(description='Convert DDP checkpoint by removing module prefix')
    parser.add_argument('--input', '-i', type=str, default='output/lbm/checkpoint.pt',
                        help='Input checkpoint path')
    parser.add_argument('--output', '-o', type=str, default='checkpoints/lbm.pt',
                        help='Output checkpoint path')
    parser.add_argument('--weights-only', action='store_true', default=False,
                        help='Load checkpoint with weights_only=True')
    
    args = parser.parse_args()
    return args

def main(args):
    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} does not exist!")
        return
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    # load checkpoint.pt
    print(f"Loading checkpoint from: {args.input}")
    ckpt = torch.load(args.input, weights_only=args.weights_only, map_location='cpu')

    # print all keys
    print("Original checkpoint keys:", ckpt.keys())

    print("Original model keys:", ckpt['model'].keys())

    # Remove 'module.' prefix from model state dict keys
    new_state_dict = {}
    converted_count = 0
    kept_count = 0
    
    for key, value in ckpt['model'].items():
        if key.startswith('module.'):
            new_key = key[7:]  # Remove 'module.' prefix (7 characters)
            new_state_dict[new_key] = value
            print(f"Converted: {key} -> {new_key}")
            converted_count += 1
        else:
            new_state_dict[key] = value
            print(f"Kept: {key}")
            kept_count += 1

    # Create new checkpoint with only model key
    new_ckpt = {'model': new_state_dict}

    print(f"\nConversion summary:")
    print(f"- Converted keys: {converted_count}")
    print(f"- Kept keys: {kept_count}")
    print(f"- Total keys: {len(new_state_dict)}")
    print("\nConverted model keys:", new_ckpt['model'].keys())
    print("New checkpoint keys:", new_ckpt.keys())

    # Save the converted checkpoint
    torch.save(new_ckpt, args.output)
    print(f"\nConverted checkpoint saved to: {args.output}")

if __name__ == '__main__':
    args = get_args()
    main(args)




