import torch.nn as nn
import pandas as pd
from tqdm import tqdm
from lstm import CustomLSTM
from torchnlp.encoders.text import SpacyEncoder
#import os
#os.system('python -m spacy download en')

train = pd.read_csv('train.tsv', sep='\t',index_col=None, header=None)
text_as_list = train[0].to_list()
encoder = SpacyEncoder(text_as_list)
encoded_texts = []
for i in tqdm(range(len(text_as_list))):
    encoded_texts.append(encoder.encode(text_as_list[i]))
lengths = [len(i) for i in tqdm(encoded_texts)]
print(encoded_texts)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(len(encoder.vocab) + 1, 32)
        self.lstm = CustomLSTM(32, 32)  # nn.LSTM(32, 32, batch_first=True)
        self.fc1 = nn.Linear(32, 2)

    def forward(self, x):
        x_ = self.embedding(x)
        x_, (h_n, c_n) = self.lstm(x_)
        x_ = (x_[:, -1, :])
        x_ = self.fc1(x_)
        return x_