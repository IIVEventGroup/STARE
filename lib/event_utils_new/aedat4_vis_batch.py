import os
import argparse
import numpy as np
import itertools

from torchvision.transforms import ToPILImage
from search import interpolation_search
from dv import AedatFile
from convert_event_img import convert_event_img_aedat


def process_single_aedat4(ae_file, fps, window, width, height, style):
    """
    process a single .aedat4 file and save as frame images
    """
    transform = ToPILImage()
    sequence = os.path.splitext(os.path.basename(ae_file))[0]
    parent_dir = os.path.dirname(ae_file)

    save_dir = os.path.join(parent_dir, f"{sequence}_w{window}ms", "VoxelGridComplex")
    os.makedirs(save_dir, exist_ok=True)

    with AedatFile(ae_file) as f:
        print(f"Processing: {ae_file}")
        events = np.hstack([packet for packet in f['events'].numpy()])
        events['timestamp'] = events['timestamp'] - events['timestamp'][0]

    count = 0
    time_right = 2 * window * 1e3
    while True:
        time_right += 1000 / fps * 1e3  # ms to us
        time_left = time_right - window * 1e3  # ms to us
        if time_right > events['timestamp'][-1]:
            break

        idx_start = interpolation_search(events['timestamp'], time_left)
        idx_end = interpolation_search(events['timestamp'], time_right)
        event_img = convert_event_img_aedat(events[idx_start:idx_end], style, height, width)

        img = transform(event_img)
        file_name = str(count).zfill(5) + '.jpg'
        img.save(os.path.join(save_dir, file_name))
        count += 1


def preprocess_all_aedat4(root_dir, fps, window, width, height, style, order: int, max_thread: int):
    """
    go through all .aedat4 files under root_dir and process them
    """
    ae_file_list = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.aedat4'):
                ae_file = os.path.join(dirpath, filename)
                ae_file_list.append(ae_file)


    for idx, ae_file in enumerate(ae_file_list):
        if idx % max_thread == order:
            print(f"Processing file {idx + 1}/{len(ae_file_list)}: {ae_file}")
            process_single_aedat4(ae_file, fps, window, width, height, style)


def main():
    parser = argparse.ArgumentParser(description='Preprocess all .aedat4 files in a directory recursively')
    parser.add_argument('--path_to_data', type=str, default='/PATH/TO/AEDAT4/FILES', help="Root directory to search for .aedat4 files")
    parser.add_argument('--fps', type=int, default=500, help='Output frame rate.')
    parser.add_argument('--window', type=float, default=2, help='Window size of each frame (ms).')
    parser.add_argument('--height', type=int, default=720, help='Height of the event frame.')
    parser.add_argument('--width', type=int, default=1280, help='Width of the event frame.')
    parser.add_argument('--style', type=str, default='VoxelGridComplex', help='Event frame style.')
    parser.add_argument('--order', type=int, default=0, help='Order of the thread to process files.')
    parser.add_argument('--max_thread', type=int, default=1, help='Maximum number')

    args = parser.parse_args()

    preprocess_all_aedat4(args.path_to_data, args.fps, args.window, args.width, args.height, args.style, args.order, args.max_thread)


if __name__ == '__main__':
    main()
