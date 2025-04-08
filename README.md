# *ShEPhERD*
This repository contains the code to train and sample from *ShEPhERD*'s diffusion generative model, which learns the joint distribution over 3D molecular structures and their shapes, electrostatics, and pharmacophores. At inference, *ShEPhERD* can be used to generate new molecules in their 3D conformations that exhibit target 3D interaction profiles.

Note that *ShEPhERD* has a sister repository, [shepherd-score](https://github.com/coleygroup/shepherd-score), that contains the code to generate/optimize conformers, extract interaction profiles, align molecules via their 3D interaction profiles, score 3D similarity, and evaluate samples from *ShEPhERD* by their validity, 3D similarity to a reference structure, etc. Both repositories are self-contained and have different installation requirements. The few dependencies on [shepherd-score](https://github.com/coleygroup/shepherd-score) that are necessary to train or to sample from *ShEPhERD* have been copied into `shepherd_score_utils/` for user convenience.

The preprint can be found on arXiv: [ShEPhERD: Diffusing shape, electrostatics, and pharmacophores for bioisosteric drug design](https://arxiv.org/abs/2411.04130)

<p align="center">
  <img width="400" src="./docs/images/shepherd_logo.svg">
</p>

<sub><sup>1</sup> **ShEPhERD**: **S**hape, **E**lectrostatics, and **Ph**armacophores **E**xplicit **R**epresentation **D**iffusion</sub>

## Table of Contents
1. [File Structure](##file-structure)
2. [Environment](##environment)
3. [Training and inference data](##training-and-inference-data)
4. [Training](##training)
5. [Inference](##inference)
6. [Evaluations](##evaluations)

## File Structure

THIS IS NO LONGER ACCURATE AFTER I (MATTHEW) REFACTORED! But still a good reference, can update if these changes ever make it to main repo

```
.
├── shepherd_score_utils/                       # dependencies from shepherd-score Github repository
├── shepherd_chkpts/                            # trained model checkpoints (from pytorch lightning)
├── paper_experiments/                          # inference scripts for all experiments in preprint
├── samples/                                    # empty dir to hold outputs from paper_experiments/*.py
├── jobs/                                       # empty dir to hold ouputs from train.py
├── conformers/                                 # conditional target structures for experiments, and (sample) training data
├── parameters/                                 # hyperparameter specifications for all models in preprint
├── model/                                      # model architecture
│   │   ├── equiformer_operations.py            # select E3NN operations from (original) Equiformer
│   │   ├── equiformer_v2_encoder.py            # slightly customized Equiformer-V2 module
│   │   └── model.py                            # module definitions and forward passes
│   ├── utils/                                  # misc. functions for forward passes
│   ├── egnn/                                   # customized re-implementation of EGNN
│   └── equiformer_v2/                          # clone of equiformer_v2 codebase, with slight modifications for shepherd
├── train.py                                    # main training script
├── lightning_module.py                         # pytorch-lightning modules and training pipeline
├── datasets.py                                 # torch_geometric dataset class (for training)
├── inference.py                                # inference functions; see Jupyter notebooks for example uses
├── RUNME_conditional_generation_MOSESaq.ipynb  # Jupyter notebook for conditional generation, using MOSES_aq P(x1,x3,x4) model
├── RUNME_unconditional_generation.ipynb        # Jupyter notebook for unconditional generation, for all models
├── environment.yml                             # conda environment requirements
└── README.md
```


## Environment

`environment.yml` contains the conda environment that we used for training and running *ShEPhERD*. 

**We** followed these steps to create a suitable conda environment, which worked on our Linux system. Please note that this exact installation procedure may depend on your system, particularly your cuda version. On MIT SuperCloud, the following works:

```
# MIT SUPERCLOUD INSTRUCTIONS

# get the ML packages necessary
module load anaconda/Python-ML-2025a

# the above gives access to standard ML packages like torch v2.5.1
# most people will need to also do e.g.
# https://pytorch.org/get-started/previous-versions/
# pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124


# https://pytorch-geometric.readthedocs.io/en/1.6.3/notes/installation.html
pip install --user torch-scatter -f https://pytorch-geometric.com/whl/torch-2.5.1+124.html
pip install --user torch-sparse -f https://pytorch-geometric.com/whl/torch-2.5.1+124.html
pip install --user torch-cluster -f https://pytorch-geometric.com/whl/torch-2.5.1+124.html
pip install --user torch-spline-conv -f https://pytorch-geometric.com/whl/torch-2.5.1+124.html
pip install --user torch-geometric

# other dependencies
pip install --user rdkit
pip install --user xtb
pip install --user open3d
```


## Training and inference data
`conformers/` contains the 3D structures of the natural products, PDB ligands, and fragments that we used in our experiments in the preprint. It also includes the 100 test-set structures from GDB-17 that we used in our conditional generation evaluations. 

`conformers/gdb/example_molblock_charges.pkl` contains *sample* training data from our *ShEPhERD*-GDB-17 training dataset.
`conformers/moses_aq/example_molblock_charges.pkl` contains *sample* training data from our *ShEPhERD*-MOSES_aq training dataset.

The full training data for both datasets (<10GB each) can be accessed from this Dropbox link: https://www.dropbox.com/scl/fo/rgn33g9kwthnjt27bsc3m/ADGt-CplyEXSU7u5MKc0aTo?rlkey=fhi74vkktpoj1irl84ehnw95h&e=1&st=wn46d6o2&dl=0


## Training
`train.py` is our main training script. It can be run from the command line by specifying a parameter file and a seed. All of our parameter files are held in `parameters/`. As an example, one may re-train the P(x1,x3,x4) model on ShEPhERD-MOSES-aq by calling:

`python train.py params_x1x3x4_diffusion_mosesaq_20240824 0`

Note that the trained checkpoints in `shepherd_chkpts/` were obtained after training each model for ~2 weeks on 2 V100 gpus.


## Inference

The simplest way to run inference is to follow the Jupyter notebooks `RUNME_unconditional_generation_MOSESaq.ipynb` and `RUNME_conditional_generation_MOSESaq.ipynb`. 

`paper_experiments/` also contain scripts that we used to run the experiments in our preprint. Each of these scripts should be copied into the parent directory (same directory as this README) before being called from the command line. Some of the scripts (`paper_experiments/run_inference_*_unconditional_*_.py`) take a few additional command-line arguments, which are detailed in those corresponding scripts by argparse commands.

The inference script now supports conditional generation of molecules that contain a superset of the target profile's pharmacophores via partial inpainting. [1/13/2025]


## Evaluations

This repository does *not* contain the code to evaluate samples from *ShEPhERD* (e.g., evaluate their validity, RMSD upon relaxation, 3D similarity to a target structure, etc). All such evaluations can be found in the sister repository: https://github.com/coleygroup/shepherd-score. These repositories were made separate so that the functions within [shepherd-score](https://github.com/coleygroup/shepherd-score) can be used for more general-purpose applications in ligand-based drug design. We also encourage others to use [shepherd-score](https://github.com/coleygroup/shepherd-score) to evaluate other 3D generative models besides *ShEPhERD*.


## License

This project is licensed under the MIT License -- see [LICENSE](./LICENSE) file for details.

## Citation
If you use or adapt *ShEPhERD* or [shepherd-score](https://github.com/coleygroup/shepherd-score) in your work, please cite us:

```bibtex
@article{adamsShEPhERD2024,
  title = {{{ShEPhERD}}: {{Diffusing}} Shape, Electrostatics, and Pharmacophores for Bioisosteric Drug Design},
  author = {Adams, Keir and Abeywardane, Kento and Fromer, Jenna and Coley, Connor W.},
  year = {2024},
  number = {arXiv:2411.04130},
  eprint = {2411.04130},
  publisher = {arXiv},
  doi = {10.48550/arXiv.2411.04130},
  archiveprefix = {arXiv}
}
```
