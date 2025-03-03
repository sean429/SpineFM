# SpineFM: Leveraging Foundation Models for Automatic Spine X-ray Segmentation

## Overview
This repository contains the code and data associated with the paper:

**"SpineFM: Leveraging Foundation Models for Automatic Spine X-ray Segmentation"**  
*Authors: Samuel Simons and Bartłomiej W. Papież*  
International Symposium of Biomedical Imaging, 2025

https://arxiv.org/abs/2411.00326

## Abstract
This paper introduces SpineFM, a novel pipeline that achieves state-of-the-art performance in the automatic segmentation and identification of vertebral bodies in cervical and lumbar spine radiographs. SpineFM leverages the regular geometry of the spine, employing a novel inductive process to sequentially infer the location of each vertebra along the spinal column. Vertebrae are segmented using Medical-SAM-Adaptor, a robust foundation model that diverges from commonly used CNN-based models.

## Installation
### Requirements
- Python 3.10.8 
- Required libraries/packages:
  ```bash
  pip install -r requirements.txt
  ```
### [Model Weights](weights/README.md)

### [Datasets](data/README.md)

## Usage
### Running the Code
To reproduce the results from the paper, you should: 
- Download either dataset (although currently NHANES II website is down)
- Download the corresponding model weights
- Ensure that the utils.py get_model() function weight file names match with your own
- Run the code:
- 
```bash
python main.py "output_directory" "weights_path" "dataset*" "data_path"
```
* either NHANES II or CSXA

[Provide additional usage examples or explanations of important scripts.]

##  Model Training
If you want to replicate this with a new dataset then I recommend getting in touch with me and I will try help. To start with some code is there for training the Mask R-CNN, ResNet and Point_Predictor models, although this hasn't been polished. For fine-tuning of the Medical-SAM-Adaptor see the original [repo](https://github.com/SuperMedIntel/Medical-SAM-Adapter?tab=readme-ov-file). I can provide extra details of the training process for this model if needed.

## Citation
If you use this code in your research, please cite our paper:
```bibtex
@article{simons2025spinefmleveragingfoundationmodels,
      title={SpineFM: Leveraging Foundation Models for Automatic Spine X-ray Segmentation}, 
      author={Samuel J. Simons and Bartłomiej W. Papież},
      year={2025},
      eprint={2411.00326},
      archivePrefix={arXiv},
      primaryClass={eess.IV},
      url={https://arxiv.org/abs/2411.00326}, 
}
```

## License
This code is released under the GPL-3.0 License. See the `LICENSE.txt` file for details.
