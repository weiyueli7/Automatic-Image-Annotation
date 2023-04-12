# Automatic-Image-Annotation

View the full [report](https://github.com/weiyueli7/Automatic-Image-Annotation/blob/main/report.pdf) for details.

---

## Description

In this project, we trained an algorithm to caption input images. This required the algorithm to identify objects in the images and match them to a corpus of text. We used PyTorch to implement multiple Recurrent Neural Network (RNN) models, including LSTM, Vanilla RNN, and a custom model (Architecture 2), to generate captions for the images in our dataset, specifically the well-known COCO Image Captioning Task. Due to limited GPU resources, we only used a portion of the original dataset for training, validation, and testing. We evaluated the performance of our models using cross entropy loss, BLUE-1, and BLUE-4 scores. Using deterministic sampling technique, we achieved a final test loss of 1.397, BLEU-1 score of 66.7%, and BLEU-4 score of 7.69%. Using stochastic sampling technique with a temperature of 0.1, we obtained a BLEU-1 score of 65.8% and BLEU-4 score of 7.7%. We also achieved a test loss of 1.480, BLEU-1 score of 65.0%, and BLEU-4 score of 8.9% using deterministic sampling technique, and a BLEU-1 score of 68.3% and BLEU-4 score of 8.9% using stochastic sampling technique with a temperature of 0.001, with the best Vanilla RNN model having a hidden size of 512 and an embedding size of 512. Furthermore, we obtained a test loss of 1.408, BLEU-1 score of 67.7%, and BLEU-4 score of 8.9% using deterministic sampling technique, and a BLEU-1 score of 67.7% and BLEU-4 score of 8.9% using stochastic sampling technique with a temperature of 0.001, with the best Architecture 2 model having a hidden size of 1024 and an embedding size of 512. From our analysis, we conclude that the hidden size can significantly impact the model's performance, while the embedding size does not have as much effect. Additionally, LSTM and Architecture 2, which have more control-ability due to the gate mechanism, perform better than Vanilla RNN. We also observed that passing in images at each time step does not improve performance when using cross entropy loss as an evaluation metric, but it does have a positive effect when using BLEU score, which is a more appropriate evaluation metric for text interpretation.

---
## Getting Started
---
### Dependencies

* `Python3`
* `PyTorch`
* `os`
* `PIL`
* `csv`
* `nltk`
* `tqdm`
* `copy`
* `json`
* `torch`
* `numpy`
* `shutil`
* `random`
* `pickle`
* `datetime`
* `torchvision`
* `pycocotools`
* `collections`

---

### Files
- `main.py`: Main driver class
- `experiment.py`: Main experiment class. Initialized based on config - takes care of training, saving stats and plots, logging and resuming experiments.
- `dataset_factory.py`: Factory to build datasets based on config
- `model_factory.py`: Factory to build models based on config
- `constants.py`: constants used across the project
- `file_utils.py`: utility functions for handling files 
- `caption_utils.py`: utility functions to generate bleu scores
- `vocab.py`: A simple Vocabulary wrapper
- `coco_dataset.py`: A simple implementation of `torch.utils.data.Dataset` the Coco Dataset
- `get_datasets.ipynb`: A helper notebook to set up the dataset in your workspace

---

### Executing program

1) Go to [DataHub](https://datahub.ucsd.edu/) and create a directory with all of the files in the startercode. Replace the `model_factory.py` and `experiment.py` with our implementations.
2) Run the `get_datasets.py`.
3) Define the configuration for your experiment. See `default.json` to see the structure and available options. You are free to modify and restructure the configuration as per your needs.
4) Open a new terminal on DataHub, go to the correct directory.
5) In the terminal, run:
```ruby
python3 main.py
```
This will execute the default configuration in `default.json`.

You may also **create** other `.json` files such as `arch2.json`, make sure to change the name of the experiment properly. To run other `.json` file such as `arch2.json`, run

```ruby
python3 main.py arch2
```

After finished executing, you will see all of the logs, plots, and models in the `experiment_data` folder. You should also see your sample good & bad graphs in the `visualization` folder.


---

## Help

It is recommended to set the `num_workers` configuration to `1` if you are running on DataHub; else, it might get easily to go out of memory.

To resume an ongoing experiment, simply run the same command again. It will load the latest stats and models and resume training pr evaluate performance.

---

## Authors

Contributors names and contact info (alphabetical order):

* Kong, Linghang
    * l3kong@ucsd.edu
* Li, Weiyue
    * wel019@ucsd.edu
* Li, Yi
    * yil115@ucsd.edu
* Hu, Shuangmu
    * shh036@ucsd.edu
* Wei, Yibo
    * y2wei@ucsd.edu

---

## Acknowledgments

We appreciate the help from Professor [Garrison W. Cottrell](https://cseweb.ucsd.edu/~gary/).