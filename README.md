# COIN: (C)hain (O)f (IN)terpretability

This serves as a toolkit with the aim of automatic interpretability using agents and sparse autoencoders for refined, in the loop, interpretations of neurons. This is the first step towards a pipeline of testing all of the neurons activating for a given input image and chaining the interpretations together to understand the full picture of why a network classified a target input as a specific class. 


This project combines the automation and agency of MAIA ( Multimodal Automated Interpretability Agent https://github.com/multimodal-interpretability/maia) with a sparse autoencoder (SAE) for better extraction of monosemanticity from individual neurons. 

For controlled generation of synthetic data for in the loop hypothesis testing we replace the diffusion based synthetic data generator in MAIA with the Unity gaming engine for more phyiscally grounded generation. 

Architecture:  
<img width="550" alt="Screenshot 2025-01-19 at 16 44 34" src="https://github.com/user-attachments/assets/798934ce-2161-48c7-8127-95160fa45bf0" />

Example:  
<img width="550" alt="Screenshot 2025-01-19 at 16 45 02" src="https://github.com/user-attachments/assets/6f3b9073-afd2-4354-9d26-87a8460605a7" />


Organisation
---

This repo has 3 branches, 
1. The main branch contains the Unity scene generation tools and assets. 
2. The Gemini branch contains the MAIA implementation with Gemini backbone. 
3. The SAE branch contains the sparse autoencoder.  


Installation instructions for Maia
---

```
python3 -m venv .Maia
source .Maia/bin/activate
python3 -m pip install --upgrade pip
pip install ipykernel
pip install tqdm
pip install torch torchvision
pip install numpy
pip install openai
pip install pandas
pip install requests
pip install diffusers
pip install git+https://github.com/davidbau/baukit
pip install scipy
pip install statsmodels
pip install matplotlib
pip install opencv-python
pip install git+https://github.com/IDEA-Research/GroundingDINO.git ## Errors on install, but registers as imported correctly
pip install segment-anything
pip install clip
pip install accelerate
```

Unity Assets
If you wish to give it a shot using the game engine you can test it using these assets:
---
Scene: https://assetstore.unity.com/packages/3d/environments/urban/demo-city-by-versatile-studio-mobile-friendly-269772 \
Cars: https://assetstore.unity.com/packages/3d/vehicles/land/hd-low-poly-racing-car-no-1201-118603

download [net-dissect](https://netdissect.csail.mit.edu/) precomputed exemplars:
```bash
bash download_exemplars.sh
```

### Quick Start ###
You can run demo experiments on individual units using ```demo.ipynb```:
\
\
Install Jupyter Notebook via pip (if Jupyter is already installed, continue to the next step)
```bash
pip install notebook
```
Launch Jupyter Notebook
```bash
jupyter notebook
```
This command will start the Jupyter Notebook server and open the Jupyter Notebook interface in your default web browser. The interface will show all the notebooks, files, and subdirectories in this repo (assuming is was initiated from the maia path). Open ```demo.ipynb``` and proceed according to the instructions.

 `demo.ipynb` now supports synthetic neurons. Follow installation instructions at `./synthetic-neurons-dataset/README.md`. After installation is done, you can define COIN to run on synthetic neurons according to the instructions in `demo.ipynb`.

### Batch experimentation ###
To run a batch of experiments, use ```main.py```:

#### Load openai api key ####
(in case you don't have an openai api-key, you can get one by following the instructions [here](https://platform.openai.com/docs/quickstart)).

Set your api-key as an environment variable (this is a bash command, look [here](https://platform.openai.com/docs/quickstart) for other OS)
```bash
export OPENAI_API_KEY='your-api-key-here'
```

#### Run COIN ####
Manually specify the model and desired units in the format ```layer#1=unit#1,unit#2... : layer#1=unit#1,unit#2...``` by calling e.g.:
```bash
python main.py --model resnet152 --unit_mode manual --units layer3=229,288:layer4=122,210
``` 
OR by loading a ```.json``` file specifying the units (see example in ```./neuron_indices/```)
```bash
python main.py --model resnet152 --unit_mode from_file --unit_file_path ./neuron_indices/
```
Adding ```--debug``` to the call will print all results to the screen.
Refer to the documentation of ```main.py``` for more configuration options.

Results are automatically saved to an html file under ```./results/``` and can be viewed in your browser by starting a local server:
```bash
python -m http.server 80
```
Once the server is up, open the html in [http://localhost:80](http://localhost:80/results/)

#### Run COIN on sythetic neurons ####

You can now run maia on synthetic neurons with ground-truth labels (see sec. 4.2 in the paper for more details).

Follow installation instructions at `./synthetic-neurons-dataset/README.md`. Then you should be able to run `main.py` on synthetic neurons by calling e.g.:
```bash
python main.py --model synthetic_neurons --unit_mode manual --units mono=1,8:or=9:and=0,2,5
``` 
(neuron indices are specified according to the neuron type: "mono", "or" and "and").

You can also use the .json file to run all synthetic neurons (or specify your own file):
```bash
python main.py --model synthetic_neurons --unit_mode from_file --unit_file_path ./neuron_indices/
```


