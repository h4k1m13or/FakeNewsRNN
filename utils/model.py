import torch
import torch.nn as nn
from torch import optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import utils.dataloader as dataloader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FakeNewsNet(nn.Module):
    def __init__(self, vocab_size=len(dataloader.text_field.vocab), hidden_size=400, num_layers=1, bi_lstm=False):
        super(FakeNewsNet, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bi_lstm = bi_lstm
        self.embedding = nn.Embedding(self.vocab_size, 256)
        self.LSTM = nn.LSTM(input_size=256, hidden_size=self.hidden_size, num_layers=self.num_layers,
                            bidirectional=self.bi_lstm, batch_first=True)
        self.drop = nn.Dropout(p=0.5)
        if bi_lstm:
            self.out = nn.Linear(2*self.hidden_size, 1)
        else:
            self.out = nn.Linear(self.hidden_size, 1)

    def forward(self, inp, input_len):
        embeded_text = self.embedding(inp)
        packed_input = pack_padded_sequence(embeded_text, input_len, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.LSTM(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)

        out_forward = output[range(len(output)), input_len - 1, :self.hidden_size]
        out_reverse = output[:, 0, self.hidden_size:]
        out_reduced = torch.cat((out_forward, out_reverse), 1)
        text_fea = self.drop(out_reduced)

        text_fea = self.out(text_fea)
        text_fea = torch.squeeze(text_fea, 1)
        text_out = torch.sigmoid(text_fea)

        return text_out


def save_checkpoint(save_path, model, optimizer, valid_loss):
    if save_path == None:
        return

    state_dict = {'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'valid_loss': valid_loss}

    torch.save(state_dict, save_path)
    print(f'Model saved to :{save_path}')


def load_checkpoint(load_path, model, optimizer):
    if load_path == None:
        return

    state_dict = torch.load(load_path, map_location=device)
    print(f'Model loaded from : {load_path}')

    model.load_state_dict(state_dict['model_state_dict'])
    optimizer.load_state_dict(state_dict['optimizer_state_dict'])

    return state_dict['valid_loss']


def save_metrics(save_path, train_loss_list, valid_loss_list, global_steps_list):
    if save_path == None:
        return

    state_dict = {'train_loss_list': train_loss_list,
                  'valid_loss_list': valid_loss_list,
                  'global_steps_list': global_steps_list}

    torch.save(state_dict, save_path)
    print(f'Model saved to: {save_path}')


def load_metrics(load_path):
    if load_path == None:
        return

    state_dict = torch.load(load_path, map_location=device)
    print(f'Model loaded from: {load_path}')

    return state_dict['train_loss_list'], state_dict['valid_loss_list'], state_dict['global_steps_list']


def predict(sentence):
    best_model = FakeNewsNet(num_layers=1, hidden_size=300, bi_lstm=True).to(device)
    optimizer = optim.Adam(best_model.parameters(), lr=0.001)
    load_checkpoint('./saved/model.pt', best_model, optimizer)
    from sacremoses import MosesTokenizer
    mt = MosesTokenizer(lang='en')
    tokenized = [tok for tok in mt.tokenize(sentence)]  # tokenize the sentence
    indexed = [dataloader.text_field.vocab.stoi[t] for t in tokenized]  # convert to integer sequence
    length = [len(indexed)]  # compute no. of words
    tensor = torch.LongTensor(indexed).to(device)  # convert to tensor
    tensor = tensor.unsqueeze(1).T  # reshape in form of batch,no. of words
    length_tensor = torch.LongTensor(length)  # convert to tensor
    prediction = best_model(tensor, length_tensor)  # prediction
    return prediction.item()
