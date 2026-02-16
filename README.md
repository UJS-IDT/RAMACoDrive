# RAMACoDrive

This repo is the official implementation of "RAMACoDrive: A Real-Time Asynchronous Framework for Cooperative Perception with Realistic V2X Communication".

## Installation

RAMACoDrive has been tested and verified on **Ubuntu 22.04**. 

Given the computational demands of **asynchronous multi-agent processes**, we strongly recommend the following hardware configuration:

*   **CPU:** 8 cores or higher.
*   **RAM:** 64GB or more.
*   **GPU:** At least 16GB VRAM recommended.
    > **Note:** This recommendation assumes CARLA runs on a **remote machine**. If you run CARLA locally alongside RAMACoDrive, VRAM will be shared. In that case, you may need to reduce the number of agents to prevent VRAM exhaustion.

To set up the environment, run:

```
conda create -n ramacodrive python=3.8 -y
conda activate ramacodrive
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia -y
cd your_RAMACoDrive_folder/
pip install -r requirements.txt
pip install assets/spconv-1.2.1-cp38-cp38-linux_x86_64.whl
cd src/ 
python setup.py build_ext --inplace
```

**Note on spconv:**
RAMACoDrive utilizes **spconv 1.2.1** for voxel feature generation (spconv 2.x is also supported).
*   If your environment matches this guide, try installing the provided `.whl` file first.
*   If the wheel installation fails, please install it from source:
    *   **spconv 1.2.1:** [traveller59/spconv v1.2.1](https://github.com/traveller59/spconv/tree/v1.2.1)
    *   **spconv 2.x:** [traveller59/spconv master](https://github.com/traveller59/spconv/tree/master)

## CARLA Setup

RAMACoDrive is built upon the CARLA Simulator. Please download **CARLA 0.9.15** and complete its installation first.

> **💡 Recommendation:** To ensure optimal performance, we strongly recommend running CARLA and RAMACoDrive on **separate machines**. This prevents computational resource contention (CPU/GPU) between the simulator and the driving algorithms.

### Configuration

To establish the connection, you need to modify the IP address in the following two files:

1.  **`src/tools/world_manager.py`**: Inside the `create_client` function.
2.  **`src/util.py`**: Inside the `connect_to_server` function.

Please set the IP address based on your deployment setup:

- **Single Machine (Local CARLA):**
  Set the IP to `"localhost"` or `"127.0.0.1"`.

- **Distributed Machine (Remote CARLA):**
  1. Note the **IP address** of the remote machine running CARLA.
  2. Ensure the remote firewall allows traffic on ports `2000` and `2001`.
  3. Set the IP in the files above to the **remote machine's IP**.

## Usage

Before running the code, please complete the following configurations.

### 1. Communication Configuration

Modify the `comm_trans_addr` field in your configuration file (`config/train.yml` or `config/test.yml`). You can choose between two modes:

*   **Option A: Unlimited Bandwidth (Local/Simple)**
    *   Set `comm_trans_addr` to the **IP address of the machine running RAMACoDrive**.
    *   In this mode, traffic is routed via the Ubuntu kernel, bypassing port limitations and bandwidth constraints.

*   **Option B: Realistic Communication (Remote/Limited)**
    To simulate real-world bandwidth constraints and latency:
    1.  **Configure IP:** Set `comm_trans_addr` to the IP address of your **Forwarding Machine**.
    2.  **Configure Ports:** Define the communication ports for each agent in the `inter_port` field under `agents` in the `.yml` file.
    3.  **Run Forwarder:** On the Forwarding Machine, ensure the `.yml` config matches, then run vlan/remote_communication.py on the Forwarding Machine.
    4.  **Limit Bandwidth:** Modify `traffic_control.sh` (check your Network Interface Card and ports), then execute it on the machine running RAMACoDrive to apply bandwidth limits.

### 2. Data Preparation

*   **Download Scenarios:**
    Please download the [Validation Scene](https://drive.google.com/file/d/1cw2BDVTst0t0Hxn3V0K4uszDnYDtJ-L2/view?usp=sharing) and [Test Scene](https://drive.google.com/file/d/136RP_dZiTyx-imEKemNbqQepp8aVphzb/view?usp=sharing), and place them in the carla_record directory.

*   **Create Custom Scenarios:**
    To record your own scenarios, use the `DataRecorder` class in `src/data/recorder_manager.py`.
    *   Insert `data_recorder.run_step(world)` inside the `run_step` function of your USS (Unified Sensing System) in train_main.py
    *   The save path can be configured within `recorder_manager.py`

### 3. Execution

**Train**

```
python train_main.py --debug train --log console -t Town01 -hy ${CONFIG_FILE}
```

Run the training script with your configuration file (without the `.yml` extension).

***Train Example***

```
python train_main.py --debug train --log console -t Town01 -hy point_pillar_baseline_partial
```

**Test**

```
python test_record.py --debug test --log console -t Town01 -hy ${CONFIG_FILE} --model_dir ${CHECKPOINT_FOLDER}
```

Run the testing script, specifying the checkpoint folder. You can add the `--show_sequence` flag to visualize the results during testing.

***Test Example***

```
python test_record.py --debug test --log console -t town01 -hy point_pillar_baseline_partial --model_dir ../logs/your_model_folder/ --show_sequence
```
