# Template Kit for RAMP challenge

[![Build status](https://github.com/ramp-kits/template-kit/actions/workflows/test.yml/badge.svg)](https://github.com/ramp-kits/template-kit/actions/workflows/test.yml)

## Introduction

 The challenge we propose consists in *generating room designs based on text descriptions*. The goal then is to create a model that can generate an image of a room that matches the description provided in text. This could be particularly valuable for interior designers, who can use such a model to visualize a room just from a description. We focus primarily on the ability to generate good quality images conditioned on the provided descriptions. Depending on the specificity of the text, we expect the model to either follow precise details or take creative liberties in generating the type of room featured—for instance, if the text does not provide any hints about it.  

 The data for this challenge is public and comes from a 2024 released paper (https://arxiv.org/abs/2407.05980) of Kassab et al. , which provides a rich dataset for Multi-Modal Interior Scene (MMIS) generation. Each image in the dataset is accompanied by a textual description and an audio recording, offering multiple sources of information for scene generation and recognition. The dataset covers a wide range of interior styles, layouts, and furnishings, capturing various design aesthetics. For this challenge, we've focused simply on images and textual descriptions of four specific interior styles: *Art-Deco, Coastal, Rustic, and Traditional, which we reorganized appropriately for our needs. After filtering out images without corresponding textual captions, the final dataset we use here includes **11,702 images for training, 1,655 for the public test set, and 1,656 for the private test set*. 

This task is particularly important in real-life applications such as *interior design and home planning*, where customers often formulate their design ideas through textual descriptions. By turning these descriptions into visual representations, we particularly simplify the room design process, promote diversity in design choices, and save time.


## Getting started

### Install

To run a submission and the notebook you will need the dependencies listed
in `requirements.txt`. You can install install the dependencies with the
following command-line:

```bash
pip install -U -r requirements.txt
```

If you are using `conda`, we provide an `environment.yml` file for similar
usage.

### Challenge description

Get started on this RAMP with the
[dedicated notebook](template_starting_kit.ipynb).

### Test a submission

The submissions need to be located in the `submissions` folder. For instance
for `my_submission`, it should be located in `submissions/my_submission`.

To run a specific submission, you can use the `ramp-test` command line:

```bash
ramp-test --submission my_submission
```

You can get more information regarding this command line:

```bash
ramp-test --help
```

### To go further

You can find more information regarding `ramp-workflow` in the
[dedicated documentation](https://paris-saclay-cds.github.io/ramp-docs/ramp-workflow/stable/using_kits.html)
