<div align="center">
    <summary>
      <h1>Safe Learning in the Real World via Adaptive Shielding with Hamilton-Jacobi Reachability</h1>
      <br>
    </summary>
</div>

# Overview
```math
\begin{align*}
    \phi(x, \pi_\theta(x)) :=
    \begin{cases}
        \pi_\theta(x) & V(x) \geq \epsilon \\
        \pi_{\text{safe}}(x) & \text{otherwise}
    \end{cases} \quad \text{(Least-Restrictive Safety Filter)}
\end{align*}
```

# Installation
```
conda env create -f environment.yml 
conda activate rhj
pip install -e .
```
## Computing Backward Reachable Tubes (BRTs)
Computing the BRTs requires [optimized_dp](https://github.com/SFU-MARS/optimized_dp) to be installed.

```
conda activate odp
pip install -e .
pip install gymnasium
python3 redexp/brts/dubins_3d.py
python3 redexp/brts/turtlebot_brt.py
```

# Simulation Training
```
python train/train_sac_lag.py \
    --config train/droq_config.py \
    --env_name=Safe-Dubins3d-{No,Bad,Good}ModelMismatch-v1 \
    # use robust HJ-CBF safety filter
    #--cbf \
    #--cbf_gamma=1.0 \
    # use lrc safety filter 
    # --lrc
    --utd_ratio=20 \
    --max_steps=250000
    --seed=0
```
# Turtlebot Traning
The method was tested on a [turtlebot2](https://www.turtlebot.com/turtlebot2/), using a local laptop for training. A VICON motion capture system was used to track the location of the robot.
[ROS1 Noetic](https://wiki.ros.org/noetic) was used to communicate between all systems.
## Setup On the Turtlebot
```
roscore
roslaunch turtlebot_bringup minimal.launch
```
### Setup On the Training Laptop
```
roslaunch vicon_bridge vicon.launch
python train_ros.py --config=train/dropq_config.py 
```
# Real World Training Videos
<ul align="center" style="list-style: none;">
<summary>Robust HJ-CBF Shielding Method</summary>
<a href="https://youtu.be/5K37a0UyW74">
  <img src="https://markdown-videos-api.jorgenkh.no/url?url=https%3A%2F%2Fyoutu.be%2F5K37a0UyW74" alt="Robust HJ-CBF Shielding Method" title="Robjust HJ-CBF Shielding Method"/>
</a>
<summary>Least-Restrictive Shielding Method</summary>
<a href="https://youtu.be/nrTOmq6MYLM">
  <img src="https://markdown-videos-api.jorgenkh.no/url?url=https%3A%2F%2Fyoutu.be%2FnrTOmq6MYLM" alt="Least-Restrictive Shielding Method" title="Least-Restrictive Shielding Method"/>
</a>
</ul>

# Acknowledgements
The RL implementations were built upon [jaxrl](https://github.com/ikostrikov/jaxrl) / [jaxrl5](https://github.com/kylestach/fastrlap-release/tree/main/jaxrl5).

The [vicon_bridge](https://github.com/ethz-asl/vicon_bridge) ROS package was used to communicate with the VICON motion capture systems.