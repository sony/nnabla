def get_args(monitor_path='tmp.monitor', max_iter=40000, model_save_path='tmp.monitor', learning_rate=1e-3, batch_size=64, weight_decay=0, n_devices=4, warmup_epoch=5):
    """
    Get command line arguments.

    Arguments set the default values of command line arguments.
    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", "-b", type=int, default=batch_size)
    parser.add_argument("--learning-rate", "-l",
                        type=float, default=learning_rate)
    parser.add_argument("--monitor-path", "-m",
                        type=str, default=monitor_path)
    parser.add_argument("--max-iter", "-i", type=int, default=max_iter)
    parser.add_argument("--val-interval", "-v", type=int, default=100)
    parser.add_argument("--val-iter", "-j", type=int, default=100)
    parser.add_argument("--weight-decay", "-w",
                        type=float, default=weight_decay)
    parser.add_argument("--device-id", "-d", type=int, default=0)
    parser.add_argument("--n-devices", "-n", type=int, default=n_devices)
    parser.add_argument("--warmup-epoch", "-e", type=int, default=warmup_epoch)
    parser.add_argument("--model-save-interval", "-s", type=int, default=1000)
    parser.add_argument("--model-save-path", "-o",
                        type=str, default=model_save_path)
    parser.add_argument('--context', '-c', type=str,
                        default=None, help="Extension path. ex) cpu, cuda.cudnn.")
    return parser.parse_args()
