import os
import argparse
import numpy as np

from search import interpolation_search
import dv_processing as dv
from dv import AedatFile
from convert_event_img import *


def cut_aedat4(ae_file, output_aedat4_file, left_offset=0, right_offset=0, width=1280, height=720):

    # Read events
    if os.path.exists(ae_file):
        with AedatFile(ae_file) as f:
            print('Processing:', ae_file)
            events = np.hstack([packet for packet in f['events'].numpy()])
            events['timestamp'] = events['timestamp'] - events['timestamp'][0]

        time_left = 0 + left_offset * 1e3  # Convert ms to us
        time_right = events['timestamp'][-1] - right_offset * 1e3  # Convert ms to us
        idx_start = interpolation_search(events['timestamp'], time_left)
        idx_end = interpolation_search(events['timestamp'], time_right)

        timestamps = events['timestamp'][idx_start:idx_end]
        x = events['x'][idx_start:idx_end]
        y = events['y'][idx_start:idx_end]
        p = events['polarity'][idx_start:idx_end]

        timestamps = timestamps - timestamps[0]  # Normalize timestamps

        num_events = len(timestamps)
        print(f"Total Events Number: {num_events}")

        camera_name = 'DVSync'
        config = dv.io.MonoCameraWriter.EventOnlyConfig(camera_name, (width, height))
        writer = dv.io.MonoCameraWriter(output_aedat4_file, config)
        event_store = dv.EventStore()

        print(f"Writing to {output_aedat4_file}...")
        for i in range(num_events):
            event_store.push_back(
                int(timestamps[i]),
                int(x[i]),
                int(y[i]),
                bool(p[i])
            )

        writer.writeEvents(event_store)
        print(f"\n Conversion completed, .aedat4 file saved to: {output_aedat4_file}")

    else:
        print('!!! Aedat4 File Not Found:', ae_file)


def main():
    parser = argparse.ArgumentParser(description='Preprocess the raw events into event frames')

    parser.add_argument('--path_to_aedat4', type=str, default="/PATH/TO/AEDAT4/FILE", help="Path to .aedat4 file")
    parser.add_argument('--output_aedat4_file', type=str, default="/OUTPUT/FILE/PATH", help="Output .aedat4 file path")
    parser.add_argument('--left_offset', type=int, default=2400, help="Left offset for the event sequence (ms)")
    parser.add_argument('--right_offset', type=int, default=600, help="Right offset for the event sequence (ms)")
    parser.add_argument('--width', type=int, default=1280, help="Width of the event frame")
    parser.add_argument('--height', type=int, default=720, help="Height of the event frame")

    args = parser.parse_args()

    cut_aedat4(args.path_to_aedat4, args.output_aedat4_file, args.left_offset, args.right_offset, args.width, args.height)


if __name__ == '__main__':
    main()
