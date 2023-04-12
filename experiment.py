################################################################################
# CSE 151B: Programming Assignment 4
# Code snippet by Ajit Kumar, Savyasachi
# Updated by Rohin
# Winter 2022
# Implemented by Linghang Kong, Weiyue Li, Yi Li, Shuangmu Hu, and Yibo Wei
################################################################################

import matplotlib.pyplot as plt
import numpy as np
import torch
from datetime import datetime
from tqdm import tqdm

from caption_utils import *
from constants import ROOT_STATS_DIR
from dataset_factory import get_datasets
from file_utils import *
from model_factory import get_model
from PIL import Image
import nltk
import copy

class Experiment(object):
    """
    Implementation of an RNN Experiment with three options of model (LSTM, Vanilla, and Arch2)
    """
    def __init__(self, name):
        """
        Constructor of an experiment object
        ---
        Parameters:
        name: the name of this experiment
        ---
        Attributes:
        __config_data: the configuration of the current experiment
        __name: name of this experiment
        __experiment_dir: the directory of this experiment
        __coco_test: COCO test object from the dataset
        __vocab: dictionary of the vocabs
        __train_loader: the dataloader for the train set
        __val_loader: the dataloader for the val set
        __test_loader: the dataloader for the test set
        __generation_config: the generation configuration
        __epochs: number of epochs (int)
        __current_epoch: the current epoch (int)
        __training_losses: a list that will stores the training loss
        __val_losses: a list that will stores the validation loss
        __least_val_losses: the smallest validation loss, default `float('inf')`
        __best_model: the best model of this experiment, default `None`
        __criterion: the criterion used for this experiment, default CrossEntropyLoss
        __optimizer: the optimizer for this experiment, default Adam
        """
        config_data = read_file_in_dir('./', name + '.json')
        if config_data is None:
            raise Exception("Configuration file doesn't exist: ", name)
        self.__config_data = config_data
        self.__name = self.__config_data['experiment_name']
        self.__experiment_dir = os.path.join(ROOT_STATS_DIR, self.__name)

        # Load Datasets
        self.__coco_test, self.__vocab, self.__train_loader, self.__val_loader, \
            self.__test_loader = get_datasets(config_data)

        # Setup Experiment
        self.__generation_config = self.__config_data['generation']
        self.__epochs = self.__config_data['experiment']['num_epochs']
        self.__current_epoch = 0
        self.__training_losses = []
        self.__val_losses = []
        self.__least_val_losses = float('inf')
        # Save the best model in this field and use this in test method.
        self.__best_model = None

        # Init Model
        self.__model = get_model(config_data, self.__vocab)

        # TODO: Set these Criterion and Optimizers Correctly
        self.__criterion = torch.nn.CrossEntropyLoss()
        self.__optimizer = torch.optim.Adam(params = self.__model.parameters(), 
                                            lr = self.__config_data['experiment']['learning_rate'])
        # initialize the model
        self.__init_model()

        # Load Experiment Data if available
        self.__load_experiment()

    # Loads the experiment data if exists to resume training from last saved checkpoint.
    def __load_experiment(self):
        """
        A method that initiates the experiment by reading the files in the correct directory
        If the directory DNE, create an directory.
        Provided by the starter-code, did not modify
        """
        os.makedirs(ROOT_STATS_DIR, exist_ok=True)

        if os.path.exists(self.__experiment_dir):
            self.__training_losses = read_file_in_dir(self.__experiment_dir, 'training_losses.txt')
            self.__val_losses = read_file_in_dir(self.__experiment_dir, 'val_losses.txt')
            self.__current_epoch = len(self.__training_losses)

            state_dict = torch.load(os.path.join(self.__experiment_dir, 'latest_model.pt'))
            self.__model.load_state_dict(state_dict['model'])
            self.__optimizer.load_state_dict(state_dict['optimizer'])

        else:
            os.makedirs(self.__experiment_dir)

    def __init_model(self):
        """
        A method that initialize the correct model to GPU
        Provided by the starter-code, did not modify
        """
        if torch.cuda.is_available():
            self.__model = self.__model.cuda().float()
            self.__criterion = self.__criterion.cuda()

    def run(self):
        """
        The main method that runs this experiment (training and validation)
        """
        early_stop_flag = 0
        start_epoch = self.__current_epoch
        # loop over the dataset multiple times
        for epoch in tqdm(range(start_epoch, self.__epochs)):
            # calculate the duration of this iteration
            start_time = datetime.now()
            self.__current_epoch = epoch
            # train the model
            train_loss = self.__train()
            # validate the model
            val_loss = self.__val()
            # record the stats with provided methods
            self.__record_stats(train_loss, val_loss)
            self.__log_epoch_stats(start_time)
            self.__save_model()
            # check for early stopping for efficiency purpose if applicable
            if self.__config_data['experiment']['early_stop'] != 0 and \
                val_loss > self.__least_val_losses:
                early_stop_flag += 1
                if early_stop_flag >= self.__config_data['experiment']['early_stop']:
                    print("Early stopped")
                    break
            else:
                early_stop_flag = 0

    def __train(self):
        """
        A method that trains the current model
        return the train loss (float) of this epoch
        """
        # set up the model in `train()` mode
        self.__model.train()
        training_loss = 0

        # Iterate over the data, implement the training function
        for i, (images, captions, _) in tqdm(enumerate(self.__train_loader)):
            # allocate the images and captions to GPU
            images, captions = images.cuda(), captions.cuda()
            # turn on the zero_grad
            self.__optimizer.zero_grad()
            # generate prediction with the current model
            X_train_predict = self.__model(images, captions)
            # calculate the loss
            X_train_loss = self.__criterion(X_train_predict, captions.view(-1))
            # backprop
            X_train_loss.backward()
            self.__optimizer.step()
            # update the training loss
            training_loss += X_train_loss.item()
        # return the loss of this epoch
        return training_loss / (i + 1)

    def __val(self):
        """
        A method that trains the current model
        return the validation loss (float) of this epoch
        """
        # set up the model in `eval()` mode
        self.__model.eval()
        val_loss = 0

        with torch.no_grad():
            # iterate through the validation dataloader
            for i, (images, captions, _) in enumerate(self.__val_loader):
                # allocate the images and captions to GPU
                images, captions = images.cuda(), captions.cuda()
                # generate prediction of the current model
                X_val_predict = self.__model(images, captions)
                # calculate the loss and update accordingly
                X_val_loss = self.__criterion(X_val_predict, captions.view(-1))
                val_loss += X_val_loss.item()
        # determine if updating the best model through the validation loss
        if val_loss < self.__least_val_losses:
            self.__least_val_losses = val_loss
            self.__best_model = copy.deepcopy(self.__model)
            self.__save_model('best_model.pt')
        # return the validation loss
        return val_loss / (i + 1)

    def test(self):
        """
        A method that tests the best model, generate captions and calculate bleu-1 & bleu-4 scores
        return the bleu-1 and bleu-4 scores with the test loss
        """
        # gather max_length, temperature, and deterministic from the json file
        max_length = self.__generation_config['max_length']
        temperature = self.__generation_config['temperature']
        deterministic = self.__generation_config['deterministic']
        # allow to access the best model from the directory without training this time
        if self.__best_model == None:
            self.__best_model = get_model(self.__config_data, self.__vocab)
            state_dict = torch.load(os.path.join(self.__experiment_dir, 
             'best_model.pt'))
            self.__best_model.load_state_dict(state_dict['model'])
        self.__best_model.eval()
        # initialize scores and loss
        bleu1_score_lst = []
        bleu4_score_lst = []
        good_id = []
        bad_id = []
        good_sentence = []
        bad_sentence = []
        real_good = []
        real_bad = []
        bleu_good = []
        bleu_bad = []
        good_imgs = []
        bad_imgs = []
        required_good = 3
        required_bad = 3
        test_loss = 0
        # randomize good and bad images
        rand_idx = np.random.choice(np.arange(0, 235), 30)
        
        with torch.no_grad():
            # iterate through the test dataloader
            for count, (images, captions, img_id) in tqdm(enumerate(self.__test_loader)):
                images, captions = images.cuda(), captions.cuda()
                # generate prediction with the best model and calculate the loss
                X_test_predict = self.__best_model(images, captions)
                X_test_loss = self.__criterion(X_test_predict, captions.view(-1))
                test_loss += X_test_loss.item()
                # initialize the pre_caption list
                pred_caption = []
                # generate captions from best_model's helper method
                captions = self.__best_model.generate_text(deterministic, images, \
                                                            temperature, max_length)
                # iterate through each list of captions
                for raw_caption in captions:
                    caption = []
                    # append to the captions accordingly
                    for idx in raw_caption:
                        word = self.__vocab.idx2word[idx]
                        # stop if seeing `<end>`
                        if word == '<end>':
                            break
                        elif word not in ['<start>', '<pad>']:
                            caption.append(word)
                    pred_caption.append(caption)          
                    
                # calculate the bleu score (1 or 4)
                temp_bleu1 = []
                temp_bleu4 = []
                reference = []
                # loop through current batch of images
                for i in range(len(img_id)):
                    # get ann id from image set
                    ann_ids = self.__coco_test.getAnnIds(imgIds = [img_id[i]])
                    refer_caption = []
                    # get reference caption
                    for ann_id in ann_ids:
                        caption = str(self.__coco_test.anns[ann_id]['caption'])
                        refer_caption.append(nltk.tokenize.word_tokenize(caption.lower()))
                    # calculate bleu score    
                    bl1 = bleu1(refer_caption, pred_caption[i])
                    bl4 = bleu4(refer_caption, pred_caption[i])
                    temp_bleu1.append(bl1)
                    temp_bleu4.append(bl4)
                    bleu1_score_lst.append(bl1)
                    bleu4_score_lst.append(bl4)
                    reference.append(refer_caption)
                # select the randomized chosen images    
                if count in rand_idx:
                    # find good and bad images
                    arg_max = np.argmax(temp_bleu1)
                    arg_min = np.argmin(temp_bleu1)
                    # append good images based on requirement
                    if len(pred_caption[arg_max]) > 5 and required_good > 0:
                        good_id.append(img_id[arg_max])
                        good_sentence.append(' '.join(pred_caption[arg_max]))
                        good_refer = [' '.join(refer) for refer in reference[arg_max]]
                        real_good.append(good_refer)
                        bleu_good.append((round(temp_bleu1[arg_max] / 100, 5), round(temp_bleu4[arg_max] / 100, 5)))
                        good_imgs.append(images[arg_max])
                        required_good -= 1
                    # append bad images based on requirement
                    if len(pred_caption[arg_min]) > 5 and required_bad > 0:
                        bad_id.append(img_id[arg_min])
                        bad_sentence.append(' '.join(pred_caption[arg_min]))
                        bad_refer = [' '.join(refer) for refer in reference[arg_min]]
                        real_bad.append(bad_refer)
                        bleu_bad.append((round(temp_bleu1[arg_min] / 100, 5), round(temp_bleu4[arg_min] / 100, 5)))
                        bad_imgs.append(images[arg_min])
                        required_bad -= 1
        # update the bleu_score based on the number of images in the loader and the batch_size   
        bleu1_score = np.mean(bleu1_score_lst) / 100
        bleu4_score = np.mean(bleu4_score_lst) / 100
        test_loss = test_loss / (count + 1)
        # record the performance
        result_str = "Test Performance: Loss: {}, Bleu1: {}, Bleu4: {}".format(test_loss, bleu1_score, bleu4_score)
        self.__log(result_str)
        temp = [0.001, 0.4, 5]
        self.fine_tune_visualization(good_id, bad_id, good_imgs, bad_imgs, temp, real_good, real_bad, bleu_good, bleu_bad)
            
        # return the results
        return bleu1_score, bleu4_score, good_id, bad_id, \
               good_sentence, bad_sentence, real_good, real_bad, bleu_good, bleu_bad, \
               good_imgs, bad_imgs, test_loss / (count + 1)   
    
    
    def __save_model(self, name = 'latest_model.pt'):
        """
        A helper method that save ths latest_model
        provided by the starter-code without modification
        """
        root_model_path = os.path.join(self.__experiment_dir, name)
        model_dict = self.__model.state_dict()
        state_dict = {'model': model_dict, 'optimizer': self.__optimizer.state_dict()}
        torch.save(state_dict, root_model_path)

    def __record_stats(self, train_loss, val_loss):
        """
        A helper method that record the stats
        provided by the starter-code without modification
        """
        self.__training_losses.append(train_loss)
        self.__val_losses.append(val_loss)

        self.plot_stats()

        write_to_file_in_dir(self.__experiment_dir, 'training_losses.txt', self.__training_losses)
        write_to_file_in_dir(self.__experiment_dir, 'val_losses.txt', self.__val_losses)

    def __log(self, log_str, file_name=None):
        """
        A helper method that save the stats to log
        provided by the starter-code without modification
        """
        print(log_str)
        log_to_file_in_dir(self.__experiment_dir, 'all.log', log_str)
        if file_name is not None:
            log_to_file_in_dir(self.__experiment_dir, file_name, log_str)

    def __log_epoch_stats(self, start_time):
        """
        A helper method that save the performance of epoch
        provided by the starter-code without modification
        """
        time_elapsed = datetime.now() - start_time
        time_to_completion = time_elapsed * (self.__epochs - self.__current_epoch - 1)
        train_loss = self.__training_losses[self.__current_epoch]
        val_loss = self.__val_losses[self.__current_epoch]
        summary_str = "Epoch: {}, Train Loss: {}, Val Loss: {}, Took {}, ETA: {}\n"
        summary_str = summary_str.format(self.__current_epoch + 1, train_loss, val_loss, 
                        str(time_elapsed), str(time_to_completion))
        self.__log(summary_str, 'epoch.log')
        
    def generate_path(self, ids):
        return "./data/images/test/" + "COCO_val2014_" + "0" * (12 - len(str(ids))) + str(ids) + '.jpg'
    
    def fine_tune_visualization(self, good_id, bad_id, good_images, bad_images, temperature, real_good, real_bad, bleu_good, bleu_bad):
        """
        helper method to generate caption
        
        good_id (integer list): image id of good images
        bad_id (integer list): image id of bad images
        good_images (tensor): image tensor of good images
        bad_images (tensor): image tensor of bad images
        temperature (float): required temperature
        real_good (string list): reference sentence of good images
        real_bad (string list): reference sentence of bad images
        bleu_good (float tuple list): list of bleu score for good images
        bleu_bad (float tuple list): list of bleu score for bad images
        """
        good_comment, bad_comment = [], []
        # take one image at a time
        for good_imgs in good_images:
            current_good = ""
            # repeat image in order to fit the layer of model
            good_imgs = good_imgs.unsqueeze(0).repeat(64, 1, 1, 1)
            for temp in temperature:
                good_sentence = []
                # generate captions from best_model's helper method
                captions = self.__best_model.generate_text(False, good_imgs, temp, max_length=20)
                # iterate through each list of captions
                for idx in captions[0]:
                    word = self.__vocab.idx2word[idx]
                    # stop if seeing `<end>`
                    if word == '<end>':
                        break
                    elif word not in ['<start>', '<pad>']:
                        good_sentence.append(word)
                # format text for our images
                if len(good_sentence) > 10 and good_sentence[10] != '.':
                    good_sentence = ' '.join(good_sentence[0:10]) \
                                    + '\n' + ' '.join(good_sentence[10:])
                    current_good += self.generate_comment(good_sentence, temp, False) + '\n'
                else:
                    current_good += self.generate_comment(' '.join(good_sentence), temp, False) + '\n'
            
            temp = 0.4
            captions = self.__best_model.generate_text(True, good_imgs, temp, max_length=20)
            # iterate through each list of captions
            good_sentence = []
            for idx in captions[0]:
                word = self.__vocab.idx2word[idx]
                # stop if seeing `<end>`
                if word == '<end>':
                    break
                elif word not in ['<start>', '<pad>']:
                    good_sentence.append(word)
            # format text for our images
            if len(good_sentence) > 10 and good_sentence[10] != '.':
                good_sentence = ' '.join(good_sentence[0:10]) \
                                + '\n' + ' '.join(good_sentence[10:])
                current_good += self.generate_comment(good_sentence, temp, True) + '\n'
            else:
                current_good += self.generate_comment(' '.join(good_sentence), temp, True) + '\n'
            good_comment.append(current_good)
                
        # take one image at a time
        for bad_imgs in bad_images:
            current_bad = ""
            # repeat image in order to fit the layer of model
            bad_imgs = bad_imgs.unsqueeze(0).repeat(64, 1, 1, 1)
            for temp in temperature:
                bad_sentence = []
                # generate captions from best_model's helper method
                captions = self.__best_model.generate_text(False, bad_imgs, temp, max_length=20)
                # iterate through each list of captions
                for idx in captions[0]:
                    word = self.__vocab.idx2word[idx]
                    # stop if seeing `<end>`
                    if word == '<end>':
                        break
                    elif word not in ['<start>', '<pad>']:
                        bad_sentence.append(word)
                # format text for our images        
                if len(bad_sentence) > 10 and bad_sentence[10] != '.':
                    bad_sentence = ' '.join(bad_sentence[0:10]) \
                                    + '\n' + ' '.join(bad_sentence[10:])
                    current_bad += self.generate_comment(bad_sentence, temp, False) + '\n'
                else:
                    current_bad += self.generate_comment(' '.join(bad_sentence), temp, False) + '\n'
            
            temp = 0.4
            captions = self.__best_model.generate_text(True, bad_imgs, temp, max_length=20)
            # iterate through each list of captions
            bad_sentence = []
            for idx in captions[0]:
                word = self.__vocab.idx2word[idx]
                # stop if seeing `<end>`
                if word == '<end>':
                    break
                elif word not in ['<start>', '<pad>']:
                    bad_sentence.append(word)
            # format text for our images                
            if len(bad_sentence) > 10 and bad_sentence[10] != '.':
                bad_sentence = ' '.join(bad_sentence[0:10]) \
                                + '\n' + ' '.join(bad_sentence[10:])
                current_bad += self.generate_comment(bad_sentence , temp, True) + '\n'
            else:
                current_bad += self.generate_comment(' '.join(bad_sentence), temp, True)  + '\n'
            bad_comment.append(current_bad)
        self.visualization(good_id, bad_id, good_comment, bad_comment, real_good, real_bad, bleu_good, bleu_bad)        
    
    def generate_comment(self, predict_text, temp, mode):
        """
        helper method to generate comment
        """
        # format the comment based on mode
        if mode:
            modes = 'deterministic'
            return  'When mode is: ' + modes + '\n' + "Predicted text will be: " + \
                    "\"" + predict_text + "\"" + '\n'
        else:
            modes = 'stochastic'
            return  'When temperature is: ' + str(temp) + '\n' + 'When mode is: ' \
                    + modes + '\n' + "Predicted text will be: " + "\"" + predict_text + "\"" + '\n'
    
    def visualization(self, good_id, bad_id, good_comment, bad_comment, real_good, real_bad, bleu_good, bleu_bad):
        os.makedirs("./visualization", exist_ok = True)
        """
        helper method to generate caption
        
        good_id (integer list): image id of good images
        bad_id (integer list): image id of bad images
        good_comment (string list): list of good comment
        bad_comment (string list): list of bad comment
        real_good (string list): reference sentence of good images
        real_bad (string list): reference sentence of bad images
        bleu_good (float tuple list): list of bleu score for good images
        bleu_bad (float tuple list): list of bleu score for bad images
        """
        # loop through good images
        for i in range(len(good_id)):
            comment = ""
            actual_text = ""
            # format the actual text
            for j, sent in enumerate(real_good[i]):
                actual_text += '(' + str(j+1) + '): ' + sent + '\n'
            # load images    
            path = self.generate_path(good_id[i])
            img = Image.open(path)
            titles = "Good Graph " + str(i + 1)
            plt.imshow(img)
            plt.title(titles)
            # add text for the images
            comment = good_comment[i] + '\n\n' + 'Actual text: ' + actual_text \
                                      + '\n' + 'When temperature is 0.4, bleu1 score is: ' + str(bleu_good[i][0]) \
                                      + '\n' + 'When temperature is 0.4, bleu4 score is: ' + str(bleu_good[i][1])
            file_name = "Good_Graph_" + str(i + 1) + "_" + self.__config_data["model"]["model_type"]
            plt.figtext(0, -1.35, comment, fontsize = 14)
            plt.savefig(os.path.join("./visualization", file_name + ".png"), bbox_inches = "tight")
            plt.imshow(img)
            plt.close()
            
        # loop through bad images    
        for i in range(len(bad_id)):
            comment = ""
            actual_text = ""
            # format the actual text
            for j, sent in enumerate(real_bad[i]):
                actual_text += '(' + str(j+1) + '): ' + sent + '\n'
            # load images   
            path = self.generate_path(bad_id[i])
            img = Image.open(path)
            titles = "Bad Graph " + str(i + 1)
            plt.imshow(img)
            plt.title(titles)
            # add text for the images
            comment = bad_comment[i] + '\n\n' + 'Actual text: ' + actual_text \
                                     + '\n' + 'When temperature is 0.4, bleu1 score is: ' + str(bleu_bad[i][0]) \
                                     + '\n' + 'When temperature is 0.4, bleu4 score is: ' + str(bleu_bad[i][1])
            file_name = "Bad_Graph_" + str(i + 1) + "_" + self.__config_data["model"]["model_type"]
            plt.figtext(0, -1.35, comment, fontsize = 14)
            plt.savefig(os.path.join("./visualization", file_name + ".png"), bbox_inches = "tight")
            plt.imshow(img)
            plt.close()

    def plot_stats(self):
        """
        A helper method that generates the visualization of the losses
        provided by the starter-code without modification
        """
        e = len(self.__training_losses)
        x_axis = np.arange(1, e + 1, 1)
        plt.figure()
        plt.plot(x_axis, self.__training_losses, label="Training Loss")
        plt.plot(x_axis, self.__val_losses, label="Validation Loss")
        plt.xlabel("Epochs")
        plt.legend(loc='best')
        plt.title(self.__name + " Stats Plot")
        plt.savefig(os.path.join(self.__experiment_dir, "stat_plot.png"))
        plt.show()
        plt.close()
