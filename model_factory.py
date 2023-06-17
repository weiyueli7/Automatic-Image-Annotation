import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np

# Build and return the model here based on the configuration.
class model(nn.Module):
    
    def __init__(self, model_type, embedding_size, hidden_size, vocab_size, num_layer = 2):
        
        """
        constructor of model class
        """
         
        super(model, self).__init__()
         
        # initialize variable
        self.model_type = model_type
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layer = num_layer
         
        # initialize encoder layer
        self.resnet50 = models.resnet50(pretrained = True)
        
        # freeze gradient for pretrained layers
        for layer in self.resnet50.parameters():
            layer.requires_grad_(False)
            
        # add linear layer
        self.resnet50.fc = nn.Linear(self.resnet50.fc.in_features, self.embedding_size)
        
        # initialize decoder layer
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_size)
        if self.model_type == 'LSTM':
            self.network = nn.LSTM(self.embedding_size, self.hidden_size , self.num_layer, batch_first = True)
        elif self.model_type == 'RNN':
            self.network = nn.RNN(self.embedding_size, self.hidden_size , self.num_layer, batch_first = True)
        else:
            self.network = nn.LSTM(self.embedding_size * 2, self.hidden_size , self.num_layer, batch_first = True)
            
        # initialize final linear layer
        self.linear = nn.Linear(hidden_size, self.vocab_size)
            
        
    def forward(self, image, caption):
        
        """
        image (Tensor): input vectorized image 
        caption (Tensor): input vectorized caption
        
        forward function to make predicted caption based on teaching force signals
        """
        # pass through encoder
        feature = self.resnet50(image)
        
        # pass through decoder
        embed_caption = self.embedding(caption)
        
        # delete the '<end>' input and concatenate image with caption as embeded input
        if self.model_type == 'LSTM' or self.model_type == 'RNN':
            embed_input = torch.cat((feature.unsqueeze(1), embed_caption[:, :-1, :]), dim = 1)
        else:
            # manually add '<pad>'
            pad = self.embedding(torch.zeros(image.size(0), 1, dtype = torch.long).cuda())
            embed_caption = torch.cat((pad, embed_caption[:, :-1, :]), dim = 1)
            
            # copy the image and concatenate images with each input signal caption
            feature = feature.unsqueeze(1).repeat(1, embed_caption.size(1), 1)
            embed_input = torch.cat((feature, embed_caption), dim = 2)
        
        # pass the embed input through corresponding rnn layers
        hidden_cell, memory_cell = self.network(embed_input)
        
        # pass the hidden cell through linear layer
        output = self.linear(hidden_cell)
        
        # resize the output to 2-dimension
        return output.view(-1, self.vocab_size)

    
    def generate_text(self, deterministic, images, temperature, max_length):
        
        """
        deterministic (bool): mode of generation
        images (tensor): input image
        temperature (float): temperature for stocastic
        max_length (int): max length for each sentence
        """
        # initialize softmax 
        softmax = nn.Softmax(dim = 1).cuda()
        
        # format the output captions for input images
        captions = [[] for i in range(images.shape[0])] 
        
        # initialize memory cell
        memory_cell = None
        with torch.no_grad():
            
            # pass image through encoder
            feature = self.resnet50(images)
            
            # increase dimension
            inputs = feature.unsqueeze(1)
            if self.model_type == 'ARCH2':
                
                # initialize pad
                pad = torch.zeros(feature.size(0), 1, dtype = torch.long).cuda()
                inputs = self.embedding(pad)
                
            for _ in range(max_length):
                if self.model_type == 'ARCH2':
                    
                    # concatenate image and cat
                    inputs = torch.cat((feature.unsqueeze(1), inputs), dim = 2)
                    
                # pass inputs through decoder
                hidden_cell, memory_cell = self.network(inputs, memory_cell)
                
                # pass inputs through final linear layer
                output = self.linear(hidden_cell)
                
                # resize the output
                output = output.view(-1, self.vocab_size)
                if not deterministic:
                    
                    # create softmax probability distribution
                    prob_distribution = softmax(output / temperature)
                    
                    # pick word for current position based on probability distribution
                    words = [torch.multinomial(distribution, 1).tolist() for distribution in prob_distribution]
                    
                    # transpose word list
                    words = np.transpose(words).tolist()[0]
                else:
                    
                    # pick word based on max
                    words = [torch.argmax(output, dim = 1).tolist()][0]
                
                # add word for current position to each caption
                captions = [captions[i] + [words[i]] for i in range(images.shape[0])]
                
                # update the inputs as the current output
                inputs = self.embedding(torch.tensor(words, dtype=torch.long).cuda()).unsqueeze(1)
        return captions
    
        
def get_model(config_data, vocab):
    
    """
    config_data: jason file for reference
    vocab: input caption
    getter method to create model
    """
    # initialize variable
    hidden_size = config_data['model']['hidden_size']
    embedding_size = config_data['model']['embedding_size']
    model_type = config_data['model']['model_type']

    # create the model
    return model(model_type,
                 config_data['model']['embedding_size'], 
                 config_data['model']['hidden_size'],
                 len(vocab)).cuda()
