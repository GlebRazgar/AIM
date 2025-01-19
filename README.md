# COIN: (C)hain (O)f (IN)terpretability

This serves as a toolkit with the aim of automatic interpretability using agents and sparse autoencoders for refined, in the loop, interpretations of neurons. This is the first step towards a pipeline of testing all of the neurons activating for a given input image and chaining the interpretations together to understand the full picture of why a network classified a target input as a specific class. 


This project combines the automation and agency of MAIA ( Multimodal Automated Interpretability Agent https://github.com/multimodal-interpretability/maia) with a sparse autoencoder (SAE) for better extraction of monosemanticity from individual neurons. 

For controlled generation of synthetic data for in the loop hypothesis testing we replace the diffusion based synthetic data generator in MAIA with the Unity gaming engine for more phyiscally grounded generation. 

Overview:  
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
---
Scene: https://assetstore.unity.com/packages/3d/environments/urban/demo-city-by-versatile-studio-mobile-friendly-269772 \
Cars: https://assetstore.unity.com/packages/3d/vehicles/land/hd-low-poly-racing-car-no-1201-118603
