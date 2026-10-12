# When does a LiDAR detector stop relying on the object?

> [!WARNING]
> **IN PROGRESS.** `weekN_` script names and `DIS Notes and Outputs/outputs by week/` are temporary. Final version will be a shorter streamlined codebase.

A stratified study of evidence dependence in 3D detection confidence.

One public KITTI checkpoint is held fixed. The measurement is how much its confidence depends on the LiDAR points inside a predicted box versus the surrounding scan, and whether that dependence is stronger when the object's own evidence is weak (far, sparse, or small). Nothing is trained. The detector weights are not edited.

The primary run is PointPillars (`pointpillar_7728.pth`) on the KITTI val split, Car only. Official Car moderate 3D AP@R11 on the saved predictions is **77.2898** (published 77.28). Compare that R11 number only. Do not compare AP@R40 to 77.28.

## What a new machine has to fetch

The git checkout contains the code. It does not contain the KITTI clouds or the PointPillars weights. Those stay out of git on purpose. Download the four archives and the one checkpoint below, then build the environment. Ubuntu 22.04 is the version this project compiles on. A newer Ubuntu (the first attempt was 26.04) fails while building OpenPCDet against this CUDA 12.1 stack. On Windows, install that distro from PowerShell:

```powershell
wsl --install -d Ubuntu-22.04
```

Open the Ubuntu 22.04 terminal for every command after that. Run `nvidia-smi` first. It needs to list a GPU before the rest of this is worth doing.

Clone this repository onto the Linux filesystem, not under `/mnt/c`. Compiling on the Windows mount is the slow, fragile path.

```bash
sudo apt update
sudo apt install -y build-essential wget git unzip
git clone https://github.com/Zachary-Woo/lidar-point-reliance.git "$HOME/lidar-point-reliance"
```

The runnable code is `$HOME/lidar-point-reliance/OpenPCDet`. That folder already contains the CPU IoU fix and the measurement scripts. Do not replace it with a fresh clone of upstream OpenPCDet.

### KITTI archives

Download these four files from the [KITTI 3D object detection page](https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d). The page asks for an email address before the downloads start. These are the only KITTI files this project uses. Leave the right-camera images, the development kit, road planes, and depth maps alone.


| Zip                        | Button on that page                                                  | Size  |
| -------------------------- | -------------------------------------------------------------------- | ----- |
| `data_object_calib.zip`    | Download camera calibration matrices of object data set              | 16 MB |
| `data_object_image_2.zip`  | Download left color images of object data set                        | 12 GB |
| `data_object_label_2.zip`  | Download training labels of object data set                          | 5 MB  |
| `data_object_velodyne.zip` | Download Velodyne point clouds, if you want to use laser information | 29 GB |


On a Windows plus Ubuntu setup, download and unpack these on Windows first, inside this project's `data/` folder. That folder is gitignored. Unpack each zip into its own directory so the Windows project contains:

```text
data/data_object_calib/training/calib/
data/data_object_calib/testing/calib/
data/data_object_image_2/training/image_2/
data/data_object_image_2/testing/image_2/
data/data_object_label_2/training/label_2/
data/data_object_velodyne/training/velodyne/
data/data_object_velodyne/testing/velodyne/
```

Each zip already contains `training/` and `testing/` paths, except the labels zip, which is training only. There is no second download onto Ubuntu. WSL already sees the Windows project under `/mnt/c/`. From the Ubuntu terminal, link those unpacked folders into the Linux checkout. `data/kitti` has to stay where it is, because it already holds `ImageSets`. Link the four content folders into `training/` and `testing/`. Linking `data/kitti` itself onto the Ubuntu `data/kitti` directory does not replace that existing folder, and the loader keeps reading an empty tree.

```bash
WIN_DATA="/mnt/c/path/to/this/project/data"
KITTI="$HOME/lidar-point-reliance/OpenPCDet/data/kitti"
mkdir -p "$KITTI/training" "$KITTI/testing"
ln -sfn "$WIN_DATA/data_object_calib/training/calib" "$KITTI/training/calib"
ln -sfn "$WIN_DATA/data_object_calib/testing/calib" "$KITTI/testing/calib"
ln -sfn "$WIN_DATA/data_object_image_2/training/image_2" "$KITTI/training/image_2"
ln -sfn "$WIN_DATA/data_object_image_2/testing/image_2" "$KITTI/testing/image_2"
ln -sfn "$WIN_DATA/data_object_label_2/training/label_2" "$KITTI/training/label_2"
ln -sfn "$WIN_DATA/data_object_velodyne/training/velodyne" "$KITTI/training/velodyne"
ln -sfn "$WIN_DATA/data_object_velodyne/testing/velodyne" "$KITTI/testing/velodyne"
```

Set `WIN_DATA` to the `/mnt/c/...` path of this project's `data` folder. A machine that has no Windows side can skip the links and unzip the four archives directly in `$KITTI` instead.

The tree that has to exist afterward:

```text
OpenPCDet/data/kitti/ImageSets/          # already in the repo, including val.txt
OpenPCDet/data/kitti/training/calib/
OpenPCDet/data/kitti/training/image_2/   # 7481 png files
OpenPCDet/data/kitti/training/label_2/
OpenPCDet/data/kitti/training/velodyne/  # 7481 bin files
OpenPCDet/data/kitti/testing/calib/
OpenPCDet/data/kitti/testing/image_2/
OpenPCDet/data/kitti/testing/velodyne/
```

The measurement reads the training split. KITTI val is a subset of those 7481 frames. The info command below also walks `testing/`, so those folders have to be there too. Check the training image count before continuing:

```bash
find "$KITTI/training/image_2" -name '*.png' | wc -l
```

That count is 7481 when `image_2` landed in the right place. `assert img_file.exists()` from `create_kitti_infos` means a zip extracted somewhere else and the loader is still looking at an empty `image_2` folder.

### PointPillars checkpoint

Download the KITTI PointPillars weights from the OpenPCDet model zoo. The row is PointPillar, Car moderate 3D AP@R11 77.28, labeled model-18M:

[https://drive.google.com/file/d/1wMxWTpU1qUoY3DsCH31WJmvJxcjFXKlm/view?usp=sharing](https://drive.google.com/file/d/1wMxWTpU1qUoY3DsCH31WJmvJxcjFXKlm/view?usp=sharing)

The browser will not name the file `pointpillar_7728.pth`. On Windows, rename the download and save it in this project as `OpenPCDet/checkpoints/pointpillar_7728.pth`. That `checkpoints` folder is gitignored. From Ubuntu, link that file into the Linux checkout:

```bash
mkdir -p "$HOME/lidar-point-reliance/OpenPCDet/checkpoints"
ln -sfn "/mnt/c/path/to/this/project/OpenPCDet/checkpoints/pointpillar_7728.pth" \
  "$HOME/lidar-point-reliance/OpenPCDet/checkpoints/pointpillar_7728.pth"
```

`test.py` and the later scripts look for that filename.

## Environment

This is the stack that compiled and ran: Python 3.9, torch 2.1.2+cu121, spconv-cu121, and the CUDA 12.1 toolkit from conda. Closing the terminal does not remove the env. Later sessions only need `conda activate pcdet`. The `CUDA_HOME` exports are for the compile.

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p "$HOME/miniconda3"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda init bash
source ~/.bashrc
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
conda create -n pcdet python=3.9 -y
conda activate pcdet
pip install "numpy==1.26.4"
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
pip install spconv-cu121
conda install -c nvidia cuda-nvcc=12.1.105 cuda-cudart-dev=12.1.105 -y
CONDA_OVERRIDE_CUDA=12.1 conda install -c nvidia/label/cuda-12.1.0 cuda-libraries-dev -y
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export LIBRARY_PATH="$CONDA_PREFIX/lib:${CONDA_PREFIX}/targets/x86_64-linux/lib:${LIBRARY_PATH:-}"
export CPATH="$CONDA_PREFIX/include:${CONDA_PREFIX}/targets/x86_64-linux/include:${CPATH:-}"
nvcc --version
cd "$HOME/lidar-point-reliance/OpenPCDet"
pip install -r requirements.txt
pip install "numpy==1.26.4"
pip install av2==0.2.0
pip install "kornia==0.5.8" --no-deps
pip install -e . --no-build-isolation
python -c "import pcdet; print(pcdet.__version__); print(pcdet.__file__)"
```

`nvcc --version` should report release 12.1. The import should print a `0.6.0` version and a path under `$HOME/lidar-point-reliance/OpenPCDet/pcdet/__init__.py`. `requirements.txt` can pull a newer NumPy, which is why 1.26.4 is installed again after it. `av2` is imported when the dataset package loads. `kornia` is installed with `--no-deps` so it does not replace the CUDA 12.1 torch build.

## First run

Build the KITTI info files once. This also writes the train and test info files and a ground-truth database. The file the study reads is `kitti_infos_val.pkl`.

```bash
conda activate pcdet
cd "$HOME/lidar-point-reliance/OpenPCDet"
python -m pcdet.datasets.kitti.kitti_dataset create_kitti_infos tools/cfgs/dataset_configs/kitti_dataset.yaml
ls data/kitti/kitti_infos_val.pkl
ls checkpoints/pointpillar_7728.pth
```

Run the pretrained checkpoint on the KITTI val split. This does not train.

```bash
cd "$HOME/lidar-point-reliance/OpenPCDet/tools"
python test.py --cfg_file cfgs/kitti_models/pointpillar.yaml --batch_size 4 --ckpt "$HOME/lidar-point-reliance/OpenPCDet/checkpoints/pointpillar_7728.pth"
```

A finished run walks 3769 samples, loads the checkpoint, and writes

`OpenPCDet/output/kitti_models/pointpillar/default/eval/epoch_7728/val/default/result.pkl`.

The score to compare is Car moderate 3D AP@R11, which is **77.2898** against the published 77.28. Compare that R11 number only. Do not compare AP@R40 to 77.28. A `NumbaPerformanceWarning` about `parallel=True` is not a failure.

## Local change inside OpenPCDet

Stock KITTI eval computes rotated-box overlap with Numba CUDA. On WSL that path segfaults even when torch CUDA works. The replacement is already in this repo:

`OpenPCDet/pcdet/datasets/kitti/kitti_object_eval_python/rotate_iou.py`

The public function is still `rotate_iou_gpu_eval`. It now runs the same IoU on CPU Numba. `device_id` is accepted and ignored so existing callers stay valid. Identical boxes are treated as a full overlap, because the polygon clipper returns a bad area when two boxes share edges.

`OpenPCDet/pcdet/datasets/kitti/kitti_object_eval_python/eval.py` is unchanged. It imports `rotate_iou_gpu_eval` and uses it for the BEV overlap inside 3D matching. A checkout of this repo already has the CPU version, so there is nothing to copy over.

## Measurement scripts

These commands run on Ubuntu, from `OpenPCDet/tools`, after `conda activate pcdet`.

A run writes the full files to `OpenPCDet/output/` on that Ubuntu checkout. The Windows project does not get that folder. OpenPCDet's own `.gitignore` ignores `output`, so the full files stay out of the git repo. The next script reads them from the Ubuntu folder. The big ones stay there on purpose, including `OpenPCDet/output/week3/per_detection_car.csv`.

The script then copies the short files into `DIS Notes and Outputs/outputs by week/` inside the same repo as the `OpenPCDet` it ran. Those weekly files are the ones that go into git. If Ubuntu and the Windows folder are two different checkouts, the copy lands in the Ubuntu repo. The Windows folder picks it up when that commit is pulled. The weekly files already in the Windows project were copied across from Ubuntu. Commit from whichever checkout has them.

Edits to a cloud are in memory. The scripts do not overwrite `training/velodyne` or `result.pkl`.

**Per-detection table.** `week3_per_detection_table.py` walks every KITTI val frame in the saved predictions and writes one row per Car box: sensor range `sqrt(x^2+y^2)`, interior point count on the full velodyne cloud, box size, best 3D IoU to a Car ground-truth box, and a match flag at IoU 0.5. Raw IoU is stored so later scoring can use the official 0.7 overlap without rebuilding the table.

```bash
python week3_per_detection_table.py
```

Read `DIS Notes and Outputs/outputs by week/week3-outputs/`. That folder has `per_detection_car_sample.csv`, `per_detection_car_summary.txt`, and `per_detection_car_definitions.txt`. The full table stays on Ubuntu at `OpenPCDet/output/week3/per_detection_car.csv`: 3769 frames, 33133 Car rows, 13458 matched at 3D IoU >= 0.5. Later scripts read that Ubuntu file. It is not copied into the notes folder and it is not in the git repo.

**Baseline stratification.** `week4_baseline_stratification.py` reads that table and the same `result.pkl`. Average precision is official KITTI 3D AP, with ground-truth cars outside a bin ignored. Confidence and expected calibration error are computed on the predictions. Bins are sensor range (0–20 m, 20–40 m, 40 m and farther), interior point count (0–19, 20–99, 100+), and box volume in cubic meters (under 8, 8–12, 12+).

```bash
python week4_baseline_stratification.py
```

Read `DIS Notes and Outputs/outputs by week/week4-outputs/week4_summary.txt`. The unstratified moderate AP@R11 there must still be 77.2898. Reliability diagrams are in that folder's `images/` directory: `reliability_range.png`, `reliability_num_points.png`, and `reliability_size.png`.

**Intervention harness.** `week5_intervention_harness.py` is the correctness check. It deletes a fraction of the points inside one predicted box, runs PointPillars on that cloud, and associates the new Car box to the original prediction at 3D IoU 0.5. Fractions are 0, 0.25, 0.5, 0.75, and 1. The draw seed is 6908. The script forwards a five-target smoke set, not the full val split.

```bash
python week5_intervention_harness.py
```

Read `DIS Notes and Outputs/outputs by week/week5-outputs/week5_check_log.txt`. A good run has zero failed checks, fraction 0 reproduces the stored box, and the velodyne files are unchanged.

**Pilot.** `week6_pilot.py` draws 200 high-confidence Car detections (score at least 0.7 and at least one interior point), stratified by range, and runs the same interior deletion. Retained confidence is the new score divided by the original score when the 0.5 IoU bar is met, and 0 when it is not. Score survival also requires the returned score to be at least 0.5.

```bash
python week6_pilot.py
```

Read `DIS Notes and Outputs/outputs by week/week6-outputs/week6_summary.txt`. The recorded decision on those definitions is GO.

**Full stratified run.** `week7_stratified.py` applies the locked metrics to every eligible Car (11125 rows) under three edits: interior deletion, deletion inside a same-size empty box (placebo), and deletion of points outside the box (context). This is the long job, on the order of 10^5 forwards. A restart skips a cell that is already in the results CSV. `--draw-only` writes the target list and stops. `--max-targets` is a preflight and refuses the official output directory.

```bash
python week7_stratified.py
```

Read `DIS Notes and Outputs/outputs by week/week7-outputs/`. `week7_summary.txt` has the marginal rates and `week7_table.csv` has every stratum, including the joint cells. Those files are the results of record. Do not launch this script to inspect them.