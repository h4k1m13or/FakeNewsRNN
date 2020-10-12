#FakeNewsRNN

Detect if a news article is real or not ! using pytorch, LSTM and Django  

#Usage and installation : 

1-Download the dataset from kaggle :https://www.kaggle.com/c/fake-news/
2-Put the files into Data directory.
3-install required packages: pip install -r requirements.txt
4-preprocess the dataset : python manage.py preprocess
5-train the model: 
python manage.py train [-h] [--min_freq
MIN_FREQ]
 [--embedding_output EMBEDDING_OUTPUT]
 [--hidden_size HIDDEN_SIZE] [--num_layers
NUM_LAYERS]
 [--num_epochs NUM_EPOCHS] [--batch_size
BATCH_SIZE]
 [--bi_lstm BI_LSTM] 
 
 6- run the django server : python manage.py runserver



