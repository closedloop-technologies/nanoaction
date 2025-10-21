# Tasks

Robotic VLA models are still in their infancy.

## Vision Priorities

1. Gather robotics data
   - OpenX Embodiment
   - python scripts/download_openx.py --list
   - python scripts/download_openx.py --datasets cmu_playing_with_food --local-dir=./data/openx/ --copy
2. Train a Video Encoder
   - https://chatgpt.com/c/68f5b629-e824-832b-8ee6-7bb5916e1e9a
   - YOLO+DeepSORT
   - Add depth estimation and Velocity esimation

3. Finetune the Video Encoder to align with the Language Model
4. Train a VLA that can control a robot via IsaacSim on Libero


## Next steps

0. Train nanochat on much smaller depth
   - Perhaps adding in more procedural knowledge and task decomposition
1. Explore OpenX dataset
   - datamodel for each experiment
   - statistics across dataset - num cameras + resolutions + data series + length ...
   - Fit the Timeseries encoder / decoder to the OpenX dataset - VQ-VAE style
     - Explore compression and vocab
2. Prompt Engineering
   - System Prompt - robot description and sensor config (URDF like)
   - output format (num steps)
   - User Prompt - Environment description + Main Task with subtasks (markdown checklist)
   - trailing actions
   - latest video (video + depth + velocity)
   - output (text or time series)
3. Training - mid training - openx
   - Dataset augmentation - shifting lags + masking timeseries + afine transform cameras
   - Labeling the dataset - substeps
   - Teaching it the output format and the camera controls
   - Fine tune the Yolo encoder / decoder model
   - Fine tune the language model to align with the 'video' encoder
   - Fine tune the timeseries encoder / decoder (ChatTS to start)
4. Training - fine tuning - issac sim
   - Simulate a wider distribution of robots and tasks
5. Evaluation
   - Run across benchmark sim datasets - Libero etc...
   - Real robot - in distribution and novel tasks
6. Fine tune the model on real robot data
   - Repeat evals
7. Deploy as a MCP service where an outer-loop agent can issue commands and read the state of the robot
