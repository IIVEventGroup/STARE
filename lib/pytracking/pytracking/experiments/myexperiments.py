from pytracking.evaluation import Tracker, get_dataset, trackerlist, load_stream_setting


# Note:
# range(1) means run_id=0, the tracking results will
# be saved in pytracking/output/tracking_results/{tracker_name}/{tracker_params}_{run_id}/
# e.g. pytracking/output/tracking_results/atom/default_000/
# if you set run_id=None, the tracking results will
# be saved in pytracking/output/tracking_results/{tracker_name}/{tracker_params}/
# e.g. pytracking/output/tracking_results/atom/default/

def fast_test_offline():
    trackers =  trackerlist('atom', 'default', range(1)) + \
                trackerlist('dimp', 'dimp18', range(1)) + \
                trackerlist('kys', 'default', range(1))
    dataset = get_dataset('esot_20_50')
    return trackers, dataset
    
def atom_nfs_uav():
    # Run three runs of ATOM on NFS and UAV datasets
    trackers = trackerlist('atom', 'default', range(3))

    dataset = get_dataset('nfs', 'uav')
    return trackers, dataset

def uav_test():
    # Run DiMP18, ATOM and ECO on the UAV dataset
    trackers = trackerlist('dimp', 'dimp18', range(1)) + \
               trackerlist('atom', 'default', range(1)) + \
               trackerlist('eco', 'default', range(1))

    dataset = get_dataset('uav')
    return trackers, dataset

def default_test():
    trackers =  trackerlist('atom', 'default', range(1)) + \
                trackerlist('dimp', 'dimp18', range(1)) + \
                trackerlist('kys', 'default', range(1)) + \
                trackerlist('rts', 'rts50', range(1)) + \
                trackerlist('keep_track','default',range(1)) +\
                trackerlist('dimp', 'prdimp18', range(1))
    dataset = get_dataset('esot500')
    return trackers, dataset

def default_online():
    trackers =  trackerlist('atom', 'default', range(1)) + \
                trackerlist('dimp', 'dimp18', range(1)) + \
                trackerlist('kys', 'default', range(1)) + \
                trackerlist('rts', 'rts50', range(1)) + \
                trackerlist('keep_track','default',range(1)) +\
                trackerlist('dimp', 'prdimp18', range(1))
    dataset = get_dataset('esot500s')
    return trackers, dataset

def default_offline():
    trackers =  trackerlist('atom', 'default', range(1)) + \
                trackerlist('dimp', 'dimp18', range(1)) + \
                trackerlist('kys', 'default', range(1)) + \
                trackerlist('rts', 'rts50', range(1)) + \
                trackerlist('keep_track','default',range(1)) +\
                trackerlist('dimp', 'prdimp18', range(1))
    dataset = get_dataset('esot500')
    return trackers, dataset

def esot500_offline():
    trackers =  trackerlist('atom', 'esot500', range(1)) + \
                trackerlist('dimp', 'dimp18_esot500', range(1)) + \
                trackerlist('dimp', 'prdimp18_esot500', range(1))
    dataset = get_dataset('esot500')
    return trackers, dataset

def esot500_fps_window():
    trackers =  trackerlist('atom', 'default', range(1)) + \
                trackerlist('dimp', 'dimp18', range(1)) + \
                trackerlist('kys', 'default', range(1)) + \
                trackerlist('rts', 'rts50', range(1)) + \
                trackerlist('keep_track','default',range(1)) +\
                trackerlist('dimp', 'prdimp18', range(1))
    dataset = get_dataset('esot_500_2','esot_250_2','esot_20_2','esot_500_8','esot_250_8','esot_20_8','esot_500_20','esot_250_20','esot_20_20')

    return trackers, dataset

def esot500_fps_window_fe():
    trackers =  trackerlist('atom', 'fe240', range(3)) + \
                trackerlist('dimp', 'dimp18_fe240', range(3)) + \
                trackerlist('dimp', 'prdimp18_fe240', range(3)) +\
                trackerlist('kys', 'fe240', range(3)) + \
                trackerlist('tomp','tomp50_fe240',range(3))
    dataset = get_dataset('esot_500_2','esot_250_2','esot_20_2','esot_500_8','esot_250_8','esot_20_8','esot_500_20','esot_250_20','esot_20_20')

    return trackers, dataset


