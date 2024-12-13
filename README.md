# SAMVS-CNN: Spatiotemporal-Attention-Modeling-for-Video-Summarization-Using-CNNs

<img width="1073" alt="image" src="https://github.com/user-attachments/assets/64f7cdaf-bd36-4a28-a5d8-5c62ac80983a" />


* [Model overview](https://github.com/VishalPrasanna11/SAMVS-CNN#model-overview)
* [Requirements](https://github.com/VishalPrasanna11/SAMVS-CNN#requirements)
* [Data](https://github.com/VishalPrasanna11/SAMVS-CNN#data)
* [Pre-trained models](https://github.com/VishalPrasanna11/SAMVS-CNN#pre-trained-models)
* [Training](https://github.com/VishalPrasanna11/SAMVS-CNN#training)
* [Inference](https://github.com/VishalPrasanna11/SAMVS-CNN#inference)
* [Generate summary videos](https://github.com/VishalPrasanna11/SAMVS-CNN#generate-summary-videos)
* [Citation](https://github.com/VishalPrasanna11/SAMVS-CNN#citation)
* [Acknowledgement](https://github.com/VishalPrasanna11/SAMVS-CNN#acknowledgement)

# Model overview
<img width="1054" alt="image" src="https://github.com/user-attachments/assets/a7a51ee4-3295-46cd-80db-989a81faab4f" />
 <br/>
SAMVS-CNN (Spatiotemporal-Attention-Modeling-for-Video-Summarization-Using-CNNs) proposes a new method for video summarization. It leverages Convolutional Neural Networks (CNNs) to model both spatial and temporal features in videos, combined with an attention mechanism to select the most relevant parts of the video for summarization.

# Requirements
|Ubuntu|GPU|CUDA|cuDNN|conda|python|
|:---:|:---:|:---:|:---:|:---:|:---:|
|20.04.6 LTS|NVIDIA a100 |12.1|8902|4.9.2|3.8.5|

|h5py|numpy|scipy|torch|torchvision|tqdm|
|:---:|:---:|:---:|:---:|:---:|:---:|
|3.1.0|1.19.5|1.5.2|2.2.1|0.17.1|4.61.0|

To set up the environment, run the following commands:

```bash
conda create -n CSTA python=3.8.5
conda activate CSTA
git clone https://github.com/VishalPrasanna11/SAMVS-CNN.git
cd SAMVS-CNN
pip install -r requirements.txt
```

# Data
Link: [Dataset](https://drive.google.com/drive/folders/1iGfKZxexQfOxyIaOWhfU0P687dJq_KWF?usp=drive_link) <br/>
H5py format of two benchmark video summarization preprocessed datasets (SumMe, TVSum). <br/>
You should download datasets and put them in ```data/``` directory. <br/>
The structure of the directory must be like below. <br/>

```
 ├── data
     └── eccv16_dataset_summe_google_pool5.h5
     └── eccv16_dataset_tvsum_google_pool5.h5
```

You can see the details of both datasets below. <br/>

[SumMe](https://link.springer.com/chapter/10.1007/978-3-319-10584-0_33) <br/>
[TVSum](https://openaccess.thecvf.com/content_cvpr_2015/papers/Song_TVSum_Summarizing_Web_2015_CVPR_paper.pdf) <br/>


Here’s the requested section in the README format:

```markdown
## Pre-trained Models

You can download our pre-trained weights of SAMVS-CNN. There are 5 weights for the SumMe dataset and the other 5 for the TVSum dataset (1 weight for each split). As shown in the paper, we tested everything 10 times (without fixation of seed) but only uploaded a single model as a representative for your convenience. The uploaded weight is acquired when the seed is `123456`, and the result is almost identical to our paper.

### Directory Structure

Please organize the weights as follows:

weights/
├── SumMe/
│   ├── split1.pt
│   ├── split2.pt
│   ├── split3.pt
│   ├── split4.pt
│   └── split5.pt
└── TVSum/
    ├── split1.pt
    ├── split2.pt
    ├── split3.pt
    ├── split4.pt
    └── split5.pt

- SumMe Weights**: Put 5 weights of the SumMe dataset in `weights/SumMe`
- VSum Weights**: Put 5 weights of the TVSum dataset in `weights/TVSum`
```

Let me know if you'd like to add anything else!

# Training
To train the model, run the following command:

```bash
python train.py
```

# Inference
Once the model is trained, you can run inference on new videos. To do this, use the following command:

```bash
python inference.py --model models/SumMe_model.pth --video_path /path/to/video.mp4 --output_path /path/to/output/
```

Make sure to provide the correct path to the trained model and input video.


# Generate summary videos
To generate a summary video using the trained model, use the following command:

```bash
python generate_summary.py --model models/SumMe_model.pth --video_path /path/to/video.mp4 --summary_output /path/to/summary_output/
```

This will generate a summarized version of the input video and save it to the specified output directory.

## Contributors

We would like to acknowledge the following contributors to the project:

- **Vishal Prasanna** - Graduate Student, Northeastern University, Email: [prasanna.vi@northeastern.edu](mailto:prasanna.vi@northeastern.edu)
- **Sai Srunith Silvery** - Graduate Student, Northeastern University, Contributor, Email: [silvery.s@northeastern.edu](mailto:silvery.s@northeastern.edu)
- **Chethana Benny** - Graduate Student, Northeastern University, Contributor, Email: [benny.c@northeastern.edu](mailto:benny.c@northeastern.edu)

Special thanks to all contributors for their invaluable input to the success of this project.

# Citation

```
@inproceedings{son2024csta,
  title={CSTA: CNN-based Spatiotemporal Attention for Video Summarization},
  author={Son, Jaewon and Park, Jaehun and Kim, Kwangsu},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={18847--18856},
  year={2024}
}

```
# Acknowledgements

We would like to express our gratitude to the video summarization research community for their contributions to the field. This work was made possible by the publicly available datasets provided by various researchers and organizations.

We also thank **CSTA**, particularly the authors Jaewon Son, Jaehun Park, and Kwangsu Kim, for their support and collaboration, which greatly contributed to the development and success of this project. Their work on "CSTA: CNN-based Spatiotemporal Attention for Video Summarization" has been invaluable in shaping our approach.
