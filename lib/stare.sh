#!/bin/bash

#export ESOT500_DIR='/your/path/to/ESOT500'
#export STARE_CKPTS_DIR='/your/path/to/stare_ckpts'

conda activate stare


#############################################################
## Step 1: prepare the dataset
#############################################################

# Check if the ESOT500_DIR environment variable is set
if [ -z "$ESOT500_DIR" ]; then
  echo "Error: ESOT500_DIR environment variable is not set. Please define it before running."
  echo "For example: export ESOT500_DIR=\"/your/path/to/ESOT500\""
  exit 1
fi

ln -s $ESOT500_DIR data/ESOT500

# Define the optional values for FPS and WINDOW
fps_options=(500 250 20)
window_options_l=(2 20 50 100 150)
window_options_h=(2 8 20 50)

echo "Starting ESOT500-L dataset preprocessing..."
echo "ESOT500-L Data Directory: $ESOT500_DIR/ESOT500-L"
echo "------------------------------------"

# Loop through all FPS options
for fps in "${fps_options[@]}"; do
  # Loop through all WINDOW options
  for window in "${window_options_l[@]}"; do
    echo "Running with: fps=${fps}, window=${window}"
    python lib/event_utils_new/esot500_preprocess.py --path_to_data "$ESOT500_DIR/ESOT500-L" --fps "$fps" --window "$window"
    echo "------------------------------------"
  done
done

ln -s data/ESOT500/ESOT500-L/500_w2ms data/ESOT500/ESOT500-L/500

echo "Starting ESOT500-H dataset preprocessing..."
echo "ESOT500-H Data Directory: $ESOT500_DIR/ESOT500-H"
echo "------------------------------------"

# Loop through all FPS options
for fps in "${fps_options[@]}"; do
  # Loop through all WINDOW options
  for window in "${window_options_h[@]}"; do
    echo "Running with: fps=${fps}, window=${window}"
    python lib/event_utils_new/esot500_preprocess.py --path_to_data "$ESOT500_DIR/ESOT500-H" --fps "$fps" --window "$window"
    echo "------------------------------------"
  done
done

ln -s data/ESOT500/ESOT500-H/500_w2ms data/ESOT500/ESOT500-H/500

echo "All ESOT500 data preprocessing tasks completed."


#############################################################
## Step 2: test the trackers under PyTracking
#############################################################

# go to the pytracking directory
cd lib/pytracking || { echo "Error: pytracking directory not found. Please check the path."; exit 1; }

# Create the default local file for pytracking
python -c "from pytracking.evaluation.environment import create_default_local_file; create_default_local_file()"
python -c "from ltr.admin.environment import create_default_local_file; create_default_local_file()"

# Link the checkpoints to the pytracking networks directory
ln -s $STARE_CKPTS_DIR/pytracking/atom pytracking/networks/atom
ln -s $STARE_CKPTS_DIR/pytracking/dimp pytracking/networks/dimp
ln -s $STARE_CKPTS_DIR/pytracking/egt pytracking/networks/egt
ln -s $STARE_CKPTS_DIR/pytracking/keep_track pytracking/networks/keep_track
ln -s $STARE_CKPTS_DIR/pytracking/kys pytracking/networks/kys
ln -s $STARE_CKPTS_DIR/pytracking/rts pytracking/networks/rts
ln -s $STARE_CKPTS_DIR/pytracking/tomp pytracking/networks/tomp

# Run frame-based tracking
echo "Starting frame-based tracking tests for trackers under PyTracking..."
echo "------------------------------------"
python pytracking/run_experiment.py exp_frame esot500_frame_all
echo "------------------------------------"

# special case for egt
for fps in "${fps_options[@]}"; do
  for window in "${window_options_l[@]}"; do
    setting_name="esot500_sim_frame_egt_${fps}_w${window}ms"
    python pytracking/run_experiment_streaming.py exp_frame "${setting_name}"
    python eval/streaming_eval_v5.py exp_frame "${setting_name}" --sim_frame --fps "${fps}" --window "${window}"
  done
done

for fps in "${fps_options[@]}"; do
  for window in "${window_options_h[@]}"; do
    setting_name="esot500h_sim_frame_egt_${fps}_w${window}ms"
    python pytracking/run_experiment_streaming.py exp_frame "${setting_name}"
    python eval/streaming_eval_v5.py exp_frame "${setting_name}" --sim_frame --fps "${fps}" --window "${window}"
  done
done

# Run stare tracking
echo "Starting stare tracking tests for trackers under PyTracking..."
echo "------------------------------------"

# Define the list of STARE window settings for evaluation
stare_window_options_l=(2 20 50 100 150 200)
stare_window_options_h=(2 8 20 50)

# Loop through each window setting
for window in "${stare_window_options_l[@]}"; do
  setting="esot500_stare_w${window}ms"
  echo "Running stare experiment for ${setting}..."
  # Run the stare experiment
  python pytracking/run_experiment_streaming.py exp_stare "${setting}"
  # Align the prediction with GT timestamp
  python eval/streaming_eval_v3.py exp_stare "${setting}"
  echo "------------------------------------"
done

for window in "${stare_window_options_h[@]}"; do
  setting="esot500h_stare_w${window}ms"
  echo "Running stare experiment for ${setting}..."
  # Run the stare experiment
  python pytracking/run_experiment_streaming.py exp_stare "${setting}"
  # Align the prediction with GT timestamp
  python eval/streaming_eval_v3.py exp_stare "${setting}"
  echo "------------------------------------"
done

echo "All tracking tests completed for trackers under PyTracking."


#############################################################
## Step 3: test the trackers under other frameworks
#############################################################

# go to the sotas directory
cd ../sotas || { echo "Error: sotas directory not found. Please check the path."; exit 1; }

###############################
# Step 3.1: MixFormer
###############################

cd MixFormer || { echo "Error: MixFormer directory not found. Please check the path."; exit 1; }

# Create the default local file for MixFormer
python -c "from lib.test.evaluation.environment import create_default_local_file; create_default_local_file()"
python -c "from lib.train.admin.environment import create_default_local_file; create_default_local_file()"

# Link the checkpoints to the sotas networks directory
ln -s $STARE_CKPTS_DIR/sotas/mixformer_convmae_online lib/test/networks/mixformer_convmae_online

# Run frame-based tracking
echo "Starting frame-based tracking tests with MixFormer ConvMAE Online baseline..."
echo "------------------------------------"

# Loop through each dataset name
for fps in "${fps_options[@]}"; do
  for window in "${window_options_l[@]}"; do
    dataset="esot_${fps}_${window}"
    echo "Running test for ESOT500-L dataset: ${dataset}"
    python tracking/test.py mixformer_convmae_online baseline --dataset_name "${dataset}"
    echo "------------------------------------"
  done
done

for fps in "${fps_options[@]}"; do
  for window in "${window_options_h[@]}"; do
    dataset="esoth_${fps}_${window}"
    echo "Running test for ESOT500-H dataset: ${dataset}"
    python tracking/test.py mixformer_convmae_online baseline --dataset_name "${dataset}"
    echo "------------------------------------"
  done
done

# Run stare tracking
echo "Starting stare tracking tests with MixFormer ConvMAE Online baseline..."
echo "------------------------------------"

# Define the list of setting options
setting_options_l=(s101 s102 s103 s104 s105 s106)
setting_options_h=(s101 s107 s102 s103)

# Loop through each setting option
for setting in "${setting_options_l[@]}"; do
  echo "Running test for ESOT500-L setting: ${setting}"
  python tracking/test_streaming.py mixformer_convmae_online baseline "${setting}" --dataset_name esot500s
  python tracking/streaming_eval_v4.py mixformer_convmae_online baseline "${setting}" --dataset_name esot500s
  echo "------------------------------------"
done

for setting in "${setting_options_h[@]}"; do
  echo "Running test for ESOT500-H setting: ${setting}"
  python tracking/test_streaming.py mixformer_convmae_online baseline "${setting}" --dataset_name esot500hs
  python tracking/streaming_eval_v4.py mixformer_convmae_online baseline "${setting}" --dataset_name esot500hs
  echo "------------------------------------"
done

echo "All tracking tests completed for MixFormer."

###############################
# Step 3.2: Stark
###############################

cd ../Stark || { echo "Error: Stark directory not found. Please check the path."; exit 1; }

# Create the default local file for Stark
python -c "from lib.test.evaluation.environment import create_default_local_file; create_default_local_file()"
python -c "from lib.train.admin.environment import create_default_local_file; create_default_local_file()"

# Link the checkpoints to the sotas networks directory
ln -s $STARE_CKPTS_DIR/sotas/stark_s lib/test/networks/stark_s

# Run frame-based tracking
echo "Starting frame-based tracking tests with Stark_S baseline ..."
echo "------------------------------------"

# Loop through each dataset name
for fps in "${fps_options[@]}"; do
  for window in "${window_options_l[@]}"; do
    dataset="esot_${fps}_${window}"
    echo "Running test for ESOT500-L dataset: ${dataset}"
    python tracking/test.py stark_s baseline --dataset_name "${dataset}"
    echo "------------------------------------"
  done
done

for fps in "${fps_options[@]}"; do
  for window in "${window_options_h[@]}"; do
    dataset="esoth_${fps}_${window}"
    echo "Running test for ESOT500-H dataset: ${dataset}"
    python tracking/test.py stark_s baseline --dataset_name "${dataset}"
    echo "------------------------------------"
  done
done

# Run stare tracking
echo "Starting stare tracking tests with Stark_S baseline ..."
echo "------------------------------------"

# Loop through each setting option
for setting in "${setting_options_l[@]}"; do
  echo "Running test for ESOT500-L setting: ${setting}"
  python tracking/test_streaming.py stark_s baseline "${setting}" --dataset_name esot500s
  python tracking/streaming_eval_v4.py stark_s baseline "${setting}" --dataset_name esot500s
  echo "------------------------------------"
done

for setting in "${setting_options_h[@]}"; do
  echo "Running test for ESOT500-H setting: ${setting}"
  python tracking/test_streaming.py stark_s baseline "${setting}" --dataset_name esot500hs
  python tracking/streaming_eval_v4.py stark_s baseline "${setting}" --dataset_name esot500hs
  echo "------------------------------------"
done

echo "All tracking tests completed for Stark."

###############################
# Step 3.3: HDETrack
###############################

cd ../HDETrack || { echo "Error: HDETrack directory not found. Please check the path."; exit 1; }

# Create the default local file for HDETrack
python -c "from lib.test.evaluation.environment import create_default_local_file; create_default_local_file()"
python -c "from lib.train.admin.environment import create_default_local_file; create_default_local_file()"

# Link the checkpoints to the sotas networks directory
ln -s $STARE_CKPTS_DIR/sotas/hdetrack lib/test/networks/hdetrack

# Run frame-based tracking
echo "Starting frame-based tracking tests with HDETrack baseline ..."
echo "------------------------------------"

# Loop through each dataset name
for fps in "${fps_options[@]}"; do
  for window in "${window_options_l[@]}"; do
    dataset="esot_${fps}_${window}"
    echo "Running test for ESOT500-L dataset: ${dataset}"
    python tracking/test.py hdetrack hdetrack_eventvot --dataset_name "${dataset}"
    echo "------------------------------------"
  done
done

for fps in "${fps_options[@]}"; do
  for window in "${window_options_h[@]}"; do
    dataset="esoth_${fps}_${window}"
    echo "Running test for ESOT500-H dataset: ${dataset}"
    python tracking/test.py hdetrack hdetrack_eventvot --dataset_name "${dataset}"
    echo "------------------------------------"
  done
done

# Run stare tracking
echo "Starting stare tracking tests with HDETrack baseline ..."
echo "------------------------------------"

# Loop through each setting option
for setting in "${setting_options_l[@]}"; do
  echo "Running test for ESOT500-L setting: ${setting}"
  python tracking/test_streaming.py hdetrack hdetrack_eventvot "${setting}" --dataset_name esot500s
  python tracking/streaming_eval_v4.py hdetrack hdetrack_eventvot "${setting}" --dataset_name esot500s
  echo "------------------------------------"
done

for setting in "${setting_options_h[@]}"; do
  echo "Running test for ESOT500-H setting: ${setting}"
  python tracking/test_streaming.py hdetrack hdetrack_eventvot "${setting}" --dataset_name esot500hs
  python tracking/streaming_eval_v4.py hdetrack hdetrack_eventvot "${setting}" --dataset_name esot500hs
  echo "------------------------------------"
done

echo "All tracking tests completed for HDETrack."

###############################
# Step 3.4: OSTrack
###############################

cd ../OSTrack || { echo "Error: OSTrack directory not found. Please check the path."; exit 1; }

# Create the default local file for OSTrack
python -c "from lib.test.evaluation.environment import create_default_local_file; create_default_local_file()"
python -c "from lib.train.admin.environment import create_default_local_file; create_default_local_file()"

# Link the checkpoints to the sotas networks directory
ln -s $STARE_CKPTS_DIR/sotas/ostrack lib/test/networks/ostrack

# Run frame-based tracking
echo "Starting frame-based tracking tests with OSTrack ..."
echo "------------------------------------"

# Loop through each dataset name
for fps in "${fps_options[@]}"; do
  for window in "${window_options_l[@]}"; do
    dataset="esot_${fps}_${window}"
    echo "Running test for ESOT500-L dataset: ${dataset}"
    python tracking/test.py ostrack esot500mix --dataset_name "${dataset}"
    echo "------------------------------------"
  done
done

for fps in "${fps_options[@]}"; do
  for window in "${window_options_h[@]}"; do
    dataset="esoth_${fps}_${window}"
    echo "Running test for ESOT500-H dataset: ${dataset}"
    python tracking/test.py ostrack esot500mix --dataset_name "${dataset}"
    echo "------------------------------------"
  done
done

# Run stare tracking
echo "Starting stare tracking tests with OSTrack ..."
echo "------------------------------------"

# Loop through each setting option
for setting in "${setting_options_l[@]}"; do
  echo "Running test for ESOT500-L setting: ${setting}"
  python tracking/test_streaming.py ostrack esot500_baseline "${setting}" --dataset_name esot500s
  python tracking/streaming_eval_v4.py ostrack esot500_baseline "${setting}" --dataset_name esot500s
  echo "------------------------------------"
done

for setting in "${setting_options_h[@]}"; do
  echo "Running test for ESOT500-H setting: ${setting}"
  python tracking/test_streaming.py ostrack esot500_baseline "${setting}" --dataset_name esot500hs
  python tracking/streaming_eval_v4.py ostrack esot500_baseline "${setting}" --dataset_name esot500hs
  echo "------------------------------------"
done

echo "Starting stare tracking tests with OSTrack + CAS ..."
echo "------------------------------------"

ostrack_cas_settings=(
  "s101 1 0.5 0.05 50"
  "s102 1 0.5 0.05 50"
  "s103 2 0.5 0.05 50"
  "s104 1 0.7 0.05 50"
  "s105 1 0.7 0.05 50"
  "s106 1 0.7 0.05 50"
)

# Loop through each setting option
for setting in "${ostrack_cas_settings[@]}"; do
  read -r stare_setting cas_mode den_a den_b den_d <<< "${setting}"
  echo "Running test for ESOT500-L setting: ${stare_setting}"
  python tracking/test_streaming.py ostrack esot500_baseline "${stare_setting}" --dataset_name esot500s --runid 66 --use_cas "${cas_mode}" --den_a "${den_a}" --den_b "${den_b}" --den_d "${den_d}"
  python tracking/streaming_eval_v4.py ostrack esot500_baseline "${stare_setting}" --dataset_name esot500s --runid 66
  echo "------------------------------------"
done

echo "Running test for sparse-event scenarios of ESOT500 sequences with OSTrack + CAS"
python tracking/test_streaming.py ostrack esot500_baseline s101 --dataset_name esot500s_cas
python tracking/streaming_eval_v4.py ostrack esot500_baseline s101 --dataset_name esot500s_cas
python tracking/test_streaming.py ostrack esot500_baseline s101 --dataset_name esot500s_cas --runid 66 --use_cas 1
python tracking/streaming_eval_v4.py ostrack esot500_baseline s101 --dataset_name esot500s_cas --runid 66
python tracking/test_streaming.py ostrack esot500_baseline s101 --dataset_name esot500hs_cas
python tracking/streaming_eval_v4.py ostrack esot500_baseline s101 --dataset_name esot500hs_cas
python tracking/test_streaming.py ostrack esot500_baseline s101 --dataset_name esot500hs_cas --runid 66 --use_cas 1
python tracking/streaming_eval_v4.py ostrack esot500_baseline s101 --dataset_name esot500hs_cas --runid 66
echo "------------------------------------"

echo "All stare tracking tests completed with OSTrack."

###############################
# Step 3.5: OSTrack + Pred
###############################

cd ../pred_OSTrack || { echo "Error: Directory pred_OSTrack not found. Skipping OSTrack + Pred tests."; exit 1; }

# Create the default local file for OSTrack + Pred
python -c "from lib.test.evaluation.environment import create_default_local_file; create_default_local_file()"
python -c "from lib.train.admin.environment import create_default_local_file; create_default_local_file()"

# Link the checkpoints to the sotas networks directory
ln -s $STARE_CKPTS_DIR/sotas/ostrack lib/test/networks/ostrack

# Run stare tracking
echo "Starting stare tracking tests with OSTrack + Pred ..."
echo "------------------------------------"

# Loop through each setting option
for setting in "${setting_options_l[@]}"; do
  echo "Running test for ESOT500-L setting: ${setting}"
  python tracking/test_streaming.py ostrack pred_esot500_4step "${setting}" --dataset_name esot500s --pred_next 1
  python tracking/streaming_predspeed.py ostrack pred_esot500_4step "${setting}" --dataset_name esot500s --dynamic_order 1
  echo "------------------------------------"
done

echo "All stare tracking tests completed with OSTrack + Pred."

###############################
# Step 3.6: OSTrack + Async
###############################

cd ../async_OSTrack || { echo "Error: Directory async_OSTrack not found. Skipping OSTrack + Async tests."; exit 1; }

# Create the default local file for OSTrack + Async
python -c "from lib.test.evaluation.environment import create_default_local_file; create_default_local_file()"
python -c "from lib.train.admin.environment import create_default_local_file; create_default_local_file()"

# Link the checkpoints to the sotas networks directory
ln -s $STARE_CKPTS_DIR/sotas/ostrack lib/test/networks/ostrack
ln -s $STARE_CKPTS_DIR/sotas/ostrack/mae_pretrain_vit_base.pth pretrained_models/mae_pretrain_vit_base.pth

# Run stare tracking
echo "Starting stare tracking tests with OSTrack + Async ..."
echo "------------------------------------"

# Loop through each setting option
for setting in "${setting_options_l[@]}"; do
  echo "Running test for ESOT500-L setting: ${setting} esot500_mix_async_freezing_150_w_pretrain"
  python tracking/test_streaming.py async_ostrack esot500_mix_async_freezing_150_w_pretrain "${setting}" --dataset_name esot500s
  python tracking/streaming_eval_v4.py async_ostrack esot500_mix_async_freezing_150_w_pretrain "${setting}" --dataset_name esot500s
  echo "------------------------------------"
done

for setting in "${setting_options_l[@]}"; do
  echo "Running test for ESOT500-L setting: ${setting} esot500_020_async_freezing_150_w_pretrain"
  python tracking/test_streaming.py async_ostrack esot500_020_async_freezing_150_w_pretrain "${setting}" --dataset_name esot500s
  python tracking/streaming_eval_v4.py async_ostrack esot500_020_async_freezing_150_w_pretrain "${setting}" --dataset_name esot500s
  echo "------------------------------------"
done

echo "Starting stare tracking tests with OSTrack + Async + CAS ..."
echo "------------------------------------"

async_ostrack_cas_settings=(
  "s101 1 0.6 0.05 50"
  "s102 1 0.6 0.05 50"
  "s103 2 0.5 0.05 50"
  "s104 2 0.5 0.05 50"
  "s105 2 0.5 0.05 50"
  "s106 2 0.5 0.05 50"
)

# Loop through each setting option
for setting in "${async_ostrack_cas_settings[@]}"; do
  read -r stare_setting cas_mode den_a den_b den_d <<< "${setting}"
  echo "Running test for ESOT500-L setting: ${stare_setting} esot500_mix_async_freezing_150_w_pretrain"
  python tracking/test_streaming.py async_ostrack esot500_mix_async_freezing_150_w_pretrain "${stare_setting}" --dataset_name esot500s --runid 66 --use_cas "${cas_mode}" --den_a "${den_a}" --den_b "${den_b}" --den_d "${den_d}"
  python tracking/streaming_eval_v4.py async_ostrack esot500_mix_async_freezing_150_w_pretrain "${stare_setting}" --dataset_name esot500s --runid 66
  echo "------------------------------------"
done

echo "All stare tracking tests completed with OSTrack + Async."

###############################
# Step 3.7: Mamba_FETrackV2
###############################

cd ../Mamba_FETrackV2 || { echo "Error: Mamba_FETrackV2 directory not found. Please check the path."; exit 1; }

# Since Mamba_FETrackV2 requires a separate conda environment, we create and activate it here
echo "Setting up conda environment for Mamba_FETrackV2 ..."

M_FETV2_ENV_NAME="stare_mamba_fetrack"
M_FETV2_YML_FILE="./stare_mamba_fetrack_conda_env.yml"

echo "Checking if conda environment '$M_FETV2_ENV_NAME' exists..."
if conda info --envs | grep -q "^$M_FETV2_ENV_NAME "; then
    echo "Environment '$M_FETV2_ENV_NAME' already exists. Skipping creation."
else
    echo "Environment '$M_FETV2_ENV_NAME' not found. Starting setup..."
    conda env create -f "$M_FETV2_YML_FILE" --verbose --debug
fi

conda activate "$M_FETV2_YML_FILE"

# Create the default local file for Mamba_FETrackV2
python -c "from lib.test.evaluation.environment import create_default_local_file; create_default_local_file()"
python -c "from lib.train.admin.environment import create_default_local_file; create_default_local_file()"

# Link the checkpoints to the sotas networks directory
ln -s $STARE_CKPTS_DIR/sotas/mamba_fetrack lib/test/networks/mamba_fetrack

# Run frame-based tracking
echo "Starting frame-based tracking tests with Mamba_FETrackV2 baseline ..."
echo "------------------------------------"

# Loop through each dataset name
for fps in "${fps_options[@]}"; do
  for window in "${window_options_l[@]}"; do
    dataset="esot_${fps}_${window}"
    echo "Running test for ESOT500-L dataset: ${dataset}"
    python tracking/test.py mamba_fetrack mamba_fetrack_felt --dataset_name "${dataset}"
    echo "------------------------------------"
  done
done

# Run stare tracking
echo "Starting stare tracking tests with Mamba_FETrackV2 baseline ..."
echo "------------------------------------"

# Loop through each setting option
for setting in "${setting_options_l[@]}"; do
  echo "Running test for ESOT500-L setting: ${setting}"
  python tracking/test_streaming.py mamba_fetrack mamba_fetrack_felt "${setting}" --dataset_name esot500s
  python tracking/streaming_eval_v4.py mamba_fetrack mamba_fetrack_felt "${setting}" --dataset_name esot500s
  echo "------------------------------------"
done

echo "All tracking tests completed for Mamba_FETrackV2."

echo "All tasks completed successfully."
