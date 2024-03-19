import logging
from run.run_config import args
import torch
import sys
import os
from iopath.common.file_io import g_pathmgr
import functools
import atexit
cfg = args.parse_args()

@functools.lru_cache(maxsize=None)
def _cached_log_stream(filename):
    # Use 1K buffer if writing to cloud storage.
    io = g_pathmgr.open(
        filename, "a", buffering=1024 if "://" in filename else -1
    )
    atexit.register(io.close)
    return io

def setup_logging():
    """
    Sets up the logging for multiple processes. Only enable the logging for the
    master process, and suppress logging for the non-master processes.
    """
    # Set up logging format.
    _FORMAT = "[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s"

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    plain_formatter = logging.Formatter(
        "[%(asctime)s][%(levelname)s] %(filename)s: %(lineno)3d: %(message)s",
        datefmt="%m/%d %H:%M:%S",
    )

    if torch.cuda.current_device()==0:
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(plain_formatter)
        logger.addHandler(ch)
        
        os.makedirs(os.path.join(cfg.OUTPUT_DIR,"logs"),exist_ok=True)
        filename = os.path.join(cfg.OUTPUT_DIR, "logs/stdout.log")
        fh = logging.StreamHandler(_cached_log_stream(filename))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(plain_formatter)
        logger.addHandler(fh)

def get_logger(name):
    logger = logging.getLogger(name=name)
    
    return logger