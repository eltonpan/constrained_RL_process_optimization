# Code for *Constrained Model-free Feinforcement Learning for Process Optimization*

Elton Pan, Panagiotis Petsagkourakis, Max Mowbray, Dongda Zhang, Ehecatl Antonio del Rio-Chanonaa

Centre for Process Systems Engineering, Department of Chemical Engineering, Imperial College London, UK
Centre for Process Systems Engineering, Department of Chemical Engineering, University College London, UK c Department of Chemical Engineering and Analytical Science, University of Manchester, UK

Reinforcement learning (RL) is a control approach that can handle nonlinear stochastic optimal control problems. However, despite the promise exhibited, RL has yet to see marked translation to industrial practice primarily due to its inability to satisfy state constraints. In this work we aim to address this challenge. We propose an “oracle”-assisted constrained Q-learning algorithm that guarantees the satis- faction of joint chance constraints with a high probability, which is crucial for safety critical tasks. To achieve this, constraint tightening (backoffs) are introduced and adjusted using Broyden’s method, hence making the backoffs self-tuned. This results in a methodology that can be imbued into RL algorithms to ensure constraint satisfaction. We analyze the performance of the proposed approach and compare against nonlinear model predictive control (NMPC). The favorable performance of this algorithm signifies a step towards the incorporation of RL into real world optimization and control of engineering systems, where constraints are essential.

![Alt text](/figures/flow_chart.png "overview")

![Alt text](/figures/pseudo_code.png "osda")

![Alt text](/figures/case_study1_backoff.png "shap")

![Alt text](/figures/case_study1_mpc.png "shap")

<!-- ## Setup and installation

Run the following terminal commands 

1. Clone repo to local directory

```bash
  git clone https://github.com/eltonpan/Constrained_RL.git
```

2. Set up and activate conda environment
```bash
  cd 
```
```bash
  conda env create -f env.yml
```
```bash
  conda activate 
```

3. Add conda environment to Jupyter notebook
```bash
  conda install -c anaconda ipykernel
```
```bash
  python -m ipykernel install --user --name=
```

4. Open jupyter notebooks
```bash
  jupyter notebook <notebook_name>.ipynb
```

make sure the `zeosyn` is the environment under dropdown menu `Kernel` > `Change kernel` -->

# Cite
If you use this code, please cite this paper:
```
@article{pan2021constrained,
  title={Constrained model-free reinforcement learning for process optimization},
  author={Pan, Elton and Petsagkourakis, Panagiotis and Mowbray, Max and Zhang, Dongda and del Rio-Chanona, Ehecatl Antonio},
  journal={Computers \& Chemical Engineering},
  volume={154},
  pages={107462},
  year={2021},
  publisher={Elsevier}
}
```