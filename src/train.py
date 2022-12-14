from matplotlib import image
import torch
import torch.nn as nn
import os
import pickle
import numpy as np
from torchvision import transforms
from torch.nn.utils.rnn import pack_padded_sequence
from dataloader import get_loader
from model import Encoder, Decoder
from vocab import Vocabulary

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model_path = 'models/'
    crop_size = 224
    vocab_path = 'data/vocab.pkl'
    image_dir = 'data/resized2014'
    caption_path = 'data/annotations/captions_train2014.json'
    log_step = 1
    save_step = 1000
    embed_size = 256
    hidden_size = 512
    num_layers = 1
    num_epochs = 10
    batch_size = 128
    num_workers = 0
    learning_rate = 0.001
    
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    
    transform = transforms.Compose([
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    
    data_loader = get_loader(
        image_dir,
        caption_path,
        vocab,
        transform,
        batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    encoder = Encoder(embed_size).to(device)
    decoder = Decoder(embed_size, hidden_size, len(vocab), num_layers).to(device)
    
    criterion = nn.CrossEntropyLoss()
    params = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.bn.parameters())
    optimizer = torch.optim.Adam(params, lr=learning_rate)
    
    total_step = len(data_loader)
    
    for epoch in range(num_epochs):
        for i, (images, captions, lengths) in enumerate(data_loader):
            images = images.to(device)
            captions = captions.to(device)
            targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]
    
            features = encoder(images)
            outputs = decoder(features, captions, lengths)
            loss = criterion(outputs, targets)
    
            loss = criterion(outputs, targets)
            decoder.zero_grad()
            encoder.zero_grad()
            loss.backward()
            optimizer.step()
    
            if i % log_step == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'
                    .format(epoch, num_epochs, i, total_step, loss.item(), np.exp(loss.item()))) 
    
            if (i+1) % save_step == 0:
                torch.save(decoder.state_dict(), os.path.join(
                    model_path, 'decoder-{}-{}.ckpt'.format(epoch+1, i+1)))
                torch.save(encoder.state_dict(), os.path.join(
                    model_path, 'encoder-{}-{}.ckpt'.format(epoch+1, i+1)))


