import os
import sys
import argparse
import importlib

env_path = os.path.join(os.path.dirname(__file__), '..')
if env_path not in sys.path:
    sys.path.append(env_path)

from pytracking.evaluation import get_dataset
from pytracking.evaluation.running import run_dataset, run_dataset_stream
from pytracking.evaluation import Tracker

def load_stream_setting(stream_setting):
    """Get stream_setting."""
    param_module = importlib.import_module('pytracking.stream_settings.{}'.format(stream_setting))
    params = param_module.parameters()
    return params

def run_tracker_stream(tracker_name, tracker_param, stream_setting, run_id=None, dataset_name='otb', sequence=None, debug=0, threads=0,
                visdom_info=None):
    """Run tracker on sequence or dataset.
    args:
        tracker_name: Name of tracking method.
        tracker_param: Name of parameter file.
        run_id: The run id.
        dataset_name: Name of dataset (otb, nfs, uav, tpl, vot, tn, gott, gotv, lasot).
        sequence: Sequence number or name.
        debug: Debug level.
        threads: Number of threads.
        visdom_info: Dict optionally containing 'use_visdom', 'server' and 'port' for Visdom visualization.
    """

    visdom_info = {} if visdom_info is None else visdom_info

    dataset = get_dataset(dataset_name)

    if sequence is not None:
        dataset = [dataset[sequence]]

    trackers = [Tracker(tracker_name, tracker_param, run_id)]
    stream_setting = load_stream_setting(stream_setting) # dict

    run_dataset_stream(dataset, trackers, stream_setting, debug, threads, visdom_info=visdom_info )


def main():
    parser = argparse.ArgumentParser(description='Run tracker on sequence or dataset.')
    parser.add_argument('--tracker_name', type=str, default="dimp", help='Name of tracking method.')
    parser.add_argument('--tracker_param', type=str, default="dimp18", help='Name of parameter file.')
    parser.add_argument('--stream_setting', type=str, default="s14", help='Name of stream_setting file.')
    parser.add_argument('--runid', type=int, default=None, help='The run id.')
    parser.add_argument('--dataset_name', type=str, default='esot500s', help='Name of dataset (otb, nfs, uav, tpl, vot, tn, gott, gotv, lasot).')
    parser.add_argument('--sequence', type=str, default="airplane5_s", help='Sequence number or name.')
    parser.add_argument('--debug', type=int, default=0, help='Debug level.')
    parser.add_argument('--threads', type=int, default=0, help='Number of threads.')
    parser.add_argument('--use_visdom', type=bool, default=False, help='Flag to enable visdom.')
    parser.add_argument('--visdom_server', type=str, default='127.0.0.1', help='Server for visdom.')
    parser.add_argument('--visdom_port', type=int, default=8097, help='Port for visdom.')

    args = parser.parse_args()

    try:
        seq_name = int(args.sequence)
    except:
        seq_name = args.sequence

    run_tracker_stream(args.tracker_name, args.tracker_param,args.stream_setting, args.runid, args.dataset_name, seq_name, args.debug,
                args.threads, {'use_visdom': args.use_visdom, 'server': args.visdom_server, 'port': args.visdom_port})


if __name__ == '__main__':
    main()
