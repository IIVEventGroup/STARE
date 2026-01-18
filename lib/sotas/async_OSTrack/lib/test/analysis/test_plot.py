import os
import sys
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams['figure.figsize'] = [14, 8]

work_dir = Path.cwd().parent.parent.parent
print(work_dir)
sys.path.append(str(work_dir))
print(sys.path)

from lib.test.analysis.plot_results import plot_results, print_results, print_per_sequence_results
from lib.test.evaluation import get_dataset, trackerlist

# Frame-based evaluation - SOTAs - OSTrack

dataset_name = 'esot_20_50'
trackers = []

trackers.extend(trackerlist(name='ostrack',
                            parameter_name='esot500mix',
                            dataset_name=dataset_name,
                            run_ids=None,
                            display_name='ostrack'))

dataset = get_dataset(dataset_name)

print_results(
    trackers,
    dataset,
    dataset_name,
    force_evaluation=True,
    plot_types=('success', 'prec', 'norm_prec'),
)
