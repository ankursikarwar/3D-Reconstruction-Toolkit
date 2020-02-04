#!/bin/bash
#SBATCH --account=CVIT
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --time=3-00:00:00
#SBATCH --requeue
#SBATCH --mail-user=pranay.gupta@research.iiit.ac.in
#SBATCH --mail-type=ALL


# Purpose: Get smpl and 3d points on videos
# Run: sbatch ./script.sh <path to video dir> 
# Note: If you have images see the comment near the end of the script 
# Todo: We do not know where SMPL is stored

python2.7 -m pip install --user virtualenv
python2.7 -m virtualenv env
source env/bin/activate
git clone https://github.com/Dene33/keras_Realtime_Multi-Person_Pose_Estimation.git
pip install configobj
git clone https://github.com/Dene33/hmr.git
wget https://people.eecs.berkeley.edu/~kanazawa/cachedir/hmr/models.tar.gz && tar -xf models.tar.gz
mv models hmr/
pip install numpy
pip2 install -r hmr/requirements.txt
mkdir hmr/output

mkdir hmr/output/csv
mkdir hmr/output/images
mkdir hmr/output/csv_joined
mkdir hmr/output/images/mesh
mkdir hmr/output/images/overlay
mkdir hmr/output/bvh_animation
cd keras_Realtime_Multi-Person_Pose_Estimation
bash model/get_keras_model.sh
mkdir sample_jsons
mkdir sample_videos
mkdir sample_images

cd ..
# Copy the following Files to Given Folders
cp 2d_pose_estimation.py ./keras_Realtime_Multi-Person_Pose_Estimation/
cp demo.py ./hmr/

module load cudnn/6-cuda-8.0
module load cuda/8.0
module load ffmpeg/3.4

pip2 install keras
pip2 install tensorflow-gpu==1.3.0
# transfer videos to ./keras_Realtime_Multi-Person_Pose_Estimation/sample_videos/
#cp $1 ./keras_Realtime_Multi-Person_Pose_Estimation/sample_videos/

#bash video_to_images.sh 10
#cd ..
pip2 install pandas


#bash hmr/3dpose_estimate.sh
