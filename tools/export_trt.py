import os
import argparse
import tensorrt as trt


def main(onnx_path, engine_path, max_point_num=100, use_fp16=True, verbose=False)->None:
    """ Convert ONNX model to TensorRT engine.
    Args:
        onnx_path (str): Path to the input ONNX model.
        engine_path (str): Path to save the output TensorRT engine.
        use_fp16 (bool): Whether to use FP16 precision.
        verbose (bool): Whether to enable verbose logging.
    """
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.INFO)

    builder = trt.Builder(logger)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)

    parser = trt.OnnxParser(network, logger)
    config = builder.create_builder_config()
    
    if not os.path.isfile(onnx_path):
        raise FileNotFoundError(f"ONNX file not found: {onnx_path}")
    
    print(f"[INFO] Loading ONNX file from {onnx_path}")
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            raise RuntimeError("Failed to parse ONNX file")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)  # 4GB
    
    if use_fp16:
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("[INFO] FP16 optimization enabled.")
        else:
            print("[WARNING] FP16 not supported on this platform. Proceeding with FP32.")

    profile = builder.create_optimization_profile()
    profile.set_shape("frame", min=(1, 3, 384, 512), opt=(1, 3, 384, 512), max=(1, 3, 384, 512))
    profile.set_shape("queries", min=(1, 1, 256), opt=(1, 10, 256), max=(1, max_point_num, 256))
    profile.set_shape("collision_dist", min=(1, 1, 1, 256), opt=(1, 10, 12, 256), max=(1, max_point_num, 96, 256))
    profile.set_shape("stream_dist", min=(1, 1, 1, 256), opt=(1, 10, 12, 256), max=(1, max_point_num, 96, 256))
    profile.set_shape("vis_mask", min=(1, 1, 1), opt=(1, 10, 12), max=(1, max_point_num, 96))
    profile.set_shape("mem_mask", min=(1, 1, 1), opt=(1, 10, 12), max=(1, max_point_num, 96))
    profile.set_shape("last_pos", min=(1, 2), opt=(10, 2), max=(max_point_num, 2))
    config.add_optimization_profile(profile)

    print("[INFO] Building TensorRT engine...")
    serialized_engine = builder.build_serialized_network(network, config)
    
    if serialized_engine is None:
        raise RuntimeError("Failed to build the engine.")

    print(f"[INFO] Saving engine to {engine_path}")
    with open(engine_path, "wb") as f:
        f.write(serialized_engine)
    print("[INFO] Engine export complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert ONNX to TensorRT Engine")
    parser.add_argument("--onnx", "-i", type=str, default="checkpoints/lbm.onnx", help="Path to input ONNX model file")
    parser.add_argument("--saveEngine", "-o", type=str, default="checkpoints/lbm.engine", help="Path to output TensorRT engine file")
    parser.add_argument("--max_point_num", type=int, default=100, help="Maximum number of points")
    parser.add_argument("--fp16", default=True, action="store_true", help="Enable FP16 precision mode")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    main(
        onnx_path=args.onnx,
        engine_path=args.saveEngine,
        max_point_num=args.max_point_num,
        use_fp16=args.fp16,
        verbose=args.verbose
    )