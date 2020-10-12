import torch
from django.core.management import BaseCommand
from torchtext.data import Field, TabularDataset, BucketIterator
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.optim as optim
# Evaluation

class Command(BaseCommand):
    help = 'Training the fake news RNN'

    def add_arguments(self, parser):
        parser.add_argument('--min_freq', type=int, help='Minimun word frequency to create the vocabulary', default=5)
        parser.add_argument('--embedding_output', type=int, help='embedding layer outputs', default=256)
        parser.add_argument('--hidden_size', type=int, help='LSTM / Bi-LSTM hidden size', default=300)
        parser.add_argument('--num_layers', type=int, help='LSTM / Bi-LSTM hidden layers ', default=1)
        parser.add_argument('--num_epochs', type=int, help='number of epochs ', default=10)
        parser.add_argument('--batch_size', type=int, help='Batch size', default=64)
        parser.add_argument('--bi_lstm', type=bool, help='True for BI-LSTM, False for LSTM ', default=True)

    def handle(self, *args, **kwargs):
        min_freq = kwargs['min_freq']
        batch_size = kwargs['batch_size']
        num_epochs = kwargs['num_epochs']
        embedding_output = kwargs['embedding_output']
        hidden_size = kwargs['hidden_size']
        num_layers = kwargs['num_layers']
        bi_lstm = kwargs['bi_lstm']
        self.stdout.write("Loading Dataset ... ")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.stdout.write("creating fields ...")
        # Fields

        label_field = Field(sequential=False, use_vocab=False, batch_first=True, dtype=torch.float)
        text_field = Field(tokenize='moses', lower=True, include_lengths=True, batch_first=True)
        fields = [('label', label_field), ('title', text_field), ('text', text_field), ('titletext', text_field)]

        # TabularDataset
        self.stdout.write('creating TabularDataset...')
        train, valid, test = TabularDataset.splits(path='./data/preprocessed/', train='train.csv',
                                                   validation='valid.csv', test='test.csv',
                                                   format='CSV', fields=fields, skip_header=True)

        # Iterators
        self.stdout.write("Creating iterators...")
        train_iter = BucketIterator(train, batch_size=batch_size, sort_key=lambda x: len(x.text),
                                    device=device, sort=False, sort_within_batch=True)
        valid_iter = BucketIterator(valid, batch_size=batch_size, sort_key=lambda x: len(x.text),
                                    device=device, sort=False, sort_within_batch=True)
        test_iter = BucketIterator(test, batch_size=batch_size, sort_key=lambda x: len(x.text),
                                   device=device, sort=False, sort_within_batch=True)

        # Vocabulary
        self.stdout.write("Creating vocabulary")
        text_field.build_vocab(train, min_freq=min_freq, )

        class FakeNewsNet(nn.Module):
            def __init__(self, vocab_size=len(text_field.vocab), hidden_size=300, num_layers=1, bi_lstm=True):
                super(FakeNewsNet, self).__init__()
                self.vocab_size = vocab_size
                self.hidden_size = hidden_size
                self.num_layers = num_layers
                self.bi_lstm = bi_lstm
                self.embedding = nn.Embedding(self.vocab_size, embedding_output)
                self.LSTM = nn.LSTM(input_size=embedding_output, hidden_size=self.hidden_size, num_layers=self.num_layers,
                                    bidirectional=self.bi_lstm, batch_first=True)
                self.drop = nn.Dropout(p=0.5)
                if bi_lstm:
                    self.out = nn.Linear(2 * self.hidden_size, 1)
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
            self.stdout.write(f'Model saved to :{save_path}')

        def load_checkpoint(load_path, model, optimizer):

            if load_path == None:
                return

            state_dict = torch.load(load_path, map_location=device)
            self.stdout.write(f'Model loaded from : {load_path}')

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
            self.stdout.write(f'Model saved to: {save_path}')

        def load_metrics(load_path):

            if load_path == None:
                return

            state_dict = torch.load(load_path, map_location=device)
            self.stdout.write(f'Model loaded from: {load_path}')

            return state_dict['train_loss_list'], state_dict['valid_loss_list'], state_dict['global_steps_list']

        def train(model,
                  optimizer,
                  criterion=nn.BCELoss(),
                  train_loader=train_iter,
                  valid_loader=valid_iter,
                  num_epochs=100,
                  eval_every=len(train_iter) // 2,
                  file_path='./saved',
                  best_valid_loss=float("Inf")):

            # initialize running values
            running_loss = 0.0
            valid_running_loss = 0.0
            global_step = 0
            train_loss_list = []
            valid_loss_list = []
            global_steps_list = []

            # training loop
            self.stdout.write("training ...")
            model.train()
            for epoch in range(num_epochs):
                for (labels, (title, title_len), (text, text_len), (titletext, titletext_len)), _ in train_loader:
                    labels = labels.to(device)
                    titletext = titletext.to(device)
                    titletext_len = titletext_len.to(device)
                    output = model(titletext, titletext_len)
                    loss = criterion(output, labels)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    # update running values
                    running_loss += loss.item()
                    global_step += 1

                    # evaluation step
                    if global_step % eval_every == 0:
                        model.eval()
                        with torch.no_grad():
                            # validation loop
                            for (labels, (title, title_len), (text, text_len),
                                 (titletext, titletext_len)), _ in valid_loader:
                                labels = labels.to(device)
                                titletext = titletext.to(device)
                                titletext_len = titletext_len.to(device)
                                output = model(titletext, titletext_len)

                                loss = criterion(output, labels)
                                valid_running_loss += loss.item()

                        # evaluation
                        average_train_loss = running_loss / eval_every
                        average_valid_loss = valid_running_loss / len(valid_loader)
                        train_loss_list.append(average_train_loss)
                        valid_loss_list.append(average_valid_loss)
                        global_steps_list.append(global_step)

                        # resetting running values
                        running_loss = 0.0
                        valid_running_loss = 0.0
                        model.train()

                        # self.stdout.write progress
                        self.stdout.write('Epoch [{}/{}], Step [{}/{}], Train Loss: {:.4f}, Valid Loss: {:.4f}'
                              .format(epoch + 1, num_epochs, global_step, num_epochs * len(train_loader),
                                      average_train_loss, average_valid_loss))

                        # checkpoint
                        if best_valid_loss > average_valid_loss:
                            best_valid_loss = average_valid_loss
                            save_checkpoint(file_path + '/model.pt', model, optimizer, best_valid_loss)
                            save_metrics(file_path + '/metrics.pt', train_loss_list, valid_loss_list, global_steps_list)

            save_metrics(file_path + '/metrics.pt', train_loss_list, valid_loss_list, global_steps_list)
            self.stdout.write('Finished Training!')

        model = FakeNewsNet(hidden_size=hidden_size, num_layers=num_layers, bi_lstm=bi_lstm).to(device)
        self.stdout.write(model)
        optimizer = optim.Adam(model.parameters(), lr=0.01, eps=1e-6, )

        train(model=model, optimizer=optimizer, num_epochs=num_epochs, eval_every=2)