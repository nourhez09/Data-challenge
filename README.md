# Template Kit for RAMP challenge

[![Build status](https://github.com/ramp-kits/template-kit/actions/workflows/test.yml/badge.svg)](https://github.com/ramp-kits/template-kit/actions/workflows/test.yml)

## Introduction

The data comes from the paper https://arxiv.org/abs/2407.05980. The paper provides a dataset for advanced Multi-Modal Interior Scene (MMIS) generation. Each image within the original dataset is accompanied by its corresponding textual description and an audio recording of that description, providing rich and diverse sources of information for scene generation and recognition. MMIS encompasses a wide range of interior
spaces, capturing various styles, layouts, and furnishings.

From this data, we chose only 4 types of styles: Art-Deco, Coastal, Rustic, and Traditional. Also, we removed the images that did not have textual captions. So, the final sizes are 11702 for the training set, 1655 for the public test, 1656 for the private one.

The main task of this challenge is to make conditional reconstruction of images. In our dataset images paired with their captions. Based on the image and caption the main goal to learn a model that is capable to reconstruct the original image. The provided baseline is a VAE model. The target metric is MSE. 

Initially we wanted to generate images conditionally on their captions. However, this task seemed to complicated using this dataset and given our resources. So, we decided to try simplier taks conditional reconstuction. It is also an important task. If we had a good model for image reconstruction we could use it in severeal ways like restoring, compression images and improving their quality. So, we believe that the task is still useful.



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
