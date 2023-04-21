# Tone Mapping Image Quality Assessment
This is the code provided with the paper "An Evaluation of Objective Image Quality Assessment for Thermal Infrared Video Tone Mapping" by Teutsch et al., IEEE CVPR Workshops 2020.
Here is the direct link to the paper: [click me](https://openaccess.thecvf.com/content_CVPRW_2020/html/w6/Teutsch_An_Evaluation_of_Objective_Image_Quality_Assessment_for_Thermal_Infrared_CVPRW_2020_paper.html)

The code consists of \
(1) the plots that contain the original Matplotlib files and numbers we used in the paper, \
(2) the measures that can be used to evaluate your own tone mapping operator.

The measures are taken and re-implemented from the paper "A comparative review of tone-mapping algorithms for high dynamic range video" by Eilertsen et al., Eurographics 2017. Thanks to the participation of Gabriel Eilertsen during our re-implementation, the code is very close to the original code used in his paper. However, we only support one-channel images as we work with video data acquired by thermal infrared cameras.

## Installation and Dependencies
The code is written in Python and tested under Python 3.8. Simply clone the repository and go to directory 'plots' for the original plots provided in the paper or to directory 'measures' for the Python code of the evaluation measures. \
If you want to apply the measures by yourself, then adjust the measures/config.json file to set the dataset you want to evaluate and to set the normalization values (depending on the bit depth of your HDR and LDR images).

If you want to add poisson noise to a set of images, you can use our included tool "add_poisson_noise". To use it you just open a terminal, and call the function as shown below.
For the parameters enter the folder path to the set of images you want to add noise to, and a folder path where you want the noisy images to be saved to.
If you want to add noise to non 16 bit images, you should change the 'norm_value' value inside the 'add_poisson_noise.py' file according to the bit depth of your images (for example 255 for 8 bit images)
```bash
cd measures
python ./add_poisson_noise.py 'ref_image_path' 'noisy_image_path'
```

Dependencies: see environment.yaml

*Note:* If you use your own dataset, you have to write your own dataset parser (except for the noise visibility measure). We currently only provide a parser for the FLIR Thermal Dataset. It may help you to format your dataset similarly to the FLIR Thermal Dataset.

## Deviation from the Paper
As we slightly changed some image pre-processing techniques after the final submission of the paper, the measures calculated with the code provided here slightly deviate from the numbers mentioned in Table 2.

The new numbers for the FLIR Thermal Dataset using the original 8-bit images provided by FLIR are: \
Underexposure (in %): 0.535 (instead of 0.54) \
Overexposure (in %): 0.899 (instead of 0.82) \
Loss of Global Contrast: -0.16216 (remains the same) \
Loss of Local Contrast: -0.04254 (instead of -0.046) \
Noise Visibility: n.a. \
Global Temporal Incoherence: 0.00017 (instead of 0.0002) \
Local Temporal Incoherence: 0.00481 (instead of 0.0253)

Furthermore, we changed some parts of the code for the noise visibility measure compared to the original version used when writing the paper. Hence, the numbers for noise visibility as mentioned for several tone mapping operators in the paper are likely to deviate.

## Acknowledgments
We like to thank the students Konstantina Mavrommati (RWTH Aachen) and Axel Goedrich (TH Nuernberg) for supporting us in implementing and revising the code for the noise visibility measure.

## Cite us
If you use the code, the plots, or the findings of our paper, then please cite us:

@InProceedings{TeutschCVPRW2020,
Title = {{An Evaluation of Objective Image Quality Assessment for Thermal Infrared Video Tone Mapping}},
Author = {Michael Teutsch and Simone Sedelmaier and Sebastian Moosbauer and Gabriel Eilertsen and Thomas Walter},
Booktitle = {IEEE CVPR Workshops},
Year = {2020}
}
