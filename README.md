# README
# Project 2: Continuous Control
## Udacity Deep Reinforcement Learning Nanodegree


## Introduction

This repository shows how `Project 2: Continuous Control` ("Reacher") from the Udacity Deep Reinforcement Learning Nanodegree was tackled.

The README provides some general information. The repository also contains source code as well as a report.


## Project details

![reacher](img/visualization_1.gif)

_(Udacity)_

The goal of the project is to use Deep Reinforcement Learning to teach the controller of an agent how to track a moving target (see above). The experiences of twenty independent agents are available for training the controller. Each agent has two double-jointed arms. The environment is driven by the [Unity Machine Learning Agents Toolkit](https://github.com/Unity-Technologies/ml-agents).

The state space is 33-dimensional, representing quantities such as the position, rotation, velocity and angular velocities of the arms. The action space is four-dimensional, with all values in the interval $[-1, 1]$. These values represent the torque to be applied to the two joints. For each step that an agent's hand is in the target location, a reward of 0.1 is given. Learning is episodic, and each episode always contains exactly 1001 steps.

For this project, training of a controller is considered to be successful once the average score over all twenty agents over the preceding 100 episodes is at least 30. Within an episode, an agent's score is taken to be the sum of the rewards that that agent has received over the 1001 time-steps. The fewer episodes that are required to be successful, the better.


## Setup

The following steps will create the computing environment that was used for training.

1. On AWS, spin up a p2.xlarge instance in the N. Virginia region using the Udacity AMI `ami-016ff5559334f8619`.
2. Once the instance is up and running, SSH into the instance.
3. Run the following commands to clone the appropriate Udacity repository and install some Python dependencies:

	```
	conda activate pytorch_p36

	cd ~
	mkdir -p external/udacity

	cd ~/external/udacity
	git clone https://github.com/udacity/deep-reinforcement-learning.git

	cd ~/external/udacity/deep-reinforcement-learning/python
	pip install .
	```

4. Download resources required for the environment:

	```
	# For the environment with a single agent.
	cd ~/external/udacity/deep-reinforcement-learning/p2_continuous-control
	mkdir v1 
	cd v1
	wget https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux_NoVis.zip
	unzip Reacher_Linux_NoVis.zip

	# For the environment with twenty independent agents.
	cd ~/external/udacity/deep-reinforcement-learning/p2_continuous-control
	mkdir v2
	cd v2
	wget https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux_NoVis.zip
	unzip Reacher_Linux_NoVis.zip
	```
	
	Note that for this step, 'no visualisation'-flavoured resources are used.
	
5. Install the newest version of PyTorch:

	```
	pip install torch
	```
	
6. To correct an error in Jupyter Notebook/Lab that occurs in this computing environment as of 2019-08, perform the following:

	```
	pip install 'prompt-toolkit<2.0.0,>=1.0.15' --force-reinstall
	```
	
7. Copy the files in the `src` directory of this repository to the `~/external/udacity/deep-reinforcement-learning/p2_continuous-control/src` directory on the EC2 instance.	
	
8. Deactivate the conda environment:

	```
	conda deactivate
	```
	
9. Securely set up Jupyter:

	```
	cd ~/
	mkdir ssl
	cd ssl
	openssl req -x509 -nodes -days 365 -newkey rsa:1024 -keyout "cert.key" -out "cert.pem" -batch	
	
	jupyter notebook --generate-config
	jupyter notebook password  # Enter and verify a password.
	
	```
	
10. Using an editor, add the following to the top of the `~/.jupyter/jupyter_notebook_config.py` file:

	```
	c = get_config()
	c.NotebookApp.certfile = u'/home/ubuntu/ssl/cert.pem'
	c.NotebookApp.keyfile = u'/home/ubuntu/ssl/cert.key'
	c.IPKernelApp.pylab = 'inline'
	c.NotebookApp.ip = '*'
	c.NotebookApp.open_browser = False	
	```
	
11. Start Jupyter Lab:

	```
	cd
	jupyter lab	
	```
	
12. If the EC2 security group that's in force allows traffic to port 8888, point a local browser to https://[ec2-ip-address]:8888. Otherwise execute the following in a _local_ terminal:

	```
	sudo ssh -L 443:127.0.0.1:8888 ubuntu@[ec2-ip-address]
	```
	
	Then point a local browser to https://127.0.0.1.


## Training

To replicate training, navigate to `~/external/udacity/deep-reinforcement-learning/p2_continuous-control/src` within Jupyter Lab. Open the `train.ipynb` notebook and follow the steps therein.
