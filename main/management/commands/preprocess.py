from django.core.management import BaseCommand
import pandas as pd
from sklearn.model_selection import train_test_split
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, word_tokenize
wordnet_lemmatizer = WordNetLemmatizer()

nltk.download('wordnet')
nltk.download("stopwords")
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')

class Command(BaseCommand):
    help = 'Preprocess the dataset'

    def handle(self, *args, **options):
        self.stdout.write("preprocessing started ...")
        train_set = './data/train.csv'
        test_set = 'data/test.csv'
        submit = 'data/submit.csv'
        self.stdout.write("loading dataset from ./data/")
        df_train = pd.read_csv(train_set)
        df_test = pd.read_csv(test_set)
        df_submit = pd.read_csv(submit)
        # concatinate test with submit
        # self.stdout.write(df_submit)
        df_test['label'] = df_submit['label'].values
        # self.stdout.write(df_test)

        # concatinate test with train set, we will split them later
        df = pd.concat([df_train, df_test], ignore_index=True, sort=True, ).reset_index()
        self.stdout.write("calculating NAN values")
        self.stdout.write(df.isnull().sum())  # how much null values we have for each collumn
        self.stdout.write("droping rows that contains NAN values")
        df = df.dropna()  # delete null values
        self.stdout.write(len(df))
        self.stdout.write("--------------------")
        self.stdout.write("lemmatizing and trimming texts.")
        self.stdout.write("this step takes time, please wait ....")
        def lemmatization(text):
            wnl = WordNetLemmatizer()
            punctuations = "?:!.,;"
            sentence_words = nltk.word_tokenize(text)
            lemmas = []
            for word, tag in pos_tag(word_tokenize(text)):
                wntag = tag[0].lower()
                wntag = wntag if wntag in ['a', 'r', 'n', 'v'] else None
                if not wntag:
                    lemma = word
                else:
                    lemma = wnl.lemmatize(word, wntag)
                lemmas.append(lemma)

            return " ".join(lemmas)

        def stemming(text):
            ps = PorterStemmer()
            text = re.sub('[^a-zA-Z]', ' ', text)
            text = text.lower()
            text = text.split()
            text = [ps.stem(word) for word in text if not word in stopwords.words("english")]
            text = " ".join(text)
            return text

        def trimming(text):
            text = text.split(maxsplit=600)
            text = ' '.join(text[:600])
            return text

        df['title'] = df['title'].apply(lambda x: trimming(x))
        df['title'] = df['title'].apply(lambda x: lemmatization(x))
        # df['title']=df['title'].apply(lambda x: stemming(x))
        df['text'] = df['text'].apply(lambda x: trimming(x))
        df['text'] = df['text'].apply(lambda x: lemmatization(x))
        # df['text']=df['text'].apply(lambda x: stemming(x))

        # df['titletext']=df['titletext'].apply(lambda x: lemmatization(x))
        # df['titletext']=df['titletext'].apply(lambda x: stemming(x))
        df['titletext'] = df['title'] + " . " + df['text']
        df['titletext'] = df['titletext'].apply(lambda x: trimming(x))
        df = df.reindex(columns=['label', 'title', 'text', 'titletext'])
        self.stdout.write('preprocessing finished.')
        df.drop(df[df.text.str.len() < 100].index, inplace=True)  # drop short texts
        self.stdout.write("data spliting ... ")
        # Split by label
        df_real = df[df['label'] == 0]
        df_fake = df[df['label'] == 1]

        # Train-test split
        df_real_full_train, df_real_test = train_test_split(df_real, train_size=0.7, test_size=0.3, random_state=102)
        df_fake_full_train, df_fake_test = train_test_split(df_fake, train_size=0.7, test_size=0.3, random_state=102)

        # Train-valid split
        df_real_train, df_real_valid = train_test_split(df_real_full_train, train_size=0.7, test_size=0.3,
                                                        random_state=102)
        df_fake_train, df_fake_valid = train_test_split(df_fake_full_train, train_size=0.7, test_size=0.3,
                                                        random_state=102)

        # Concatenate splits of different labels
        df_train = pd.concat([df_real_train, df_fake_train], ignore_index=True, sort=False)
        df_valid = pd.concat([df_real_valid, df_fake_valid], ignore_index=True, sort=False)
        df_test = pd.concat([df_real_test, df_fake_test], ignore_index=True, sort=False)
        self.stdout.write("saving preprocessed data to ./data/preprocessed/")
        # Write preprocessed data
        df_train.to_csv('./data/preprocessed/train.csv', index=False)
        df_valid.to_csv('./data/preprocessed/valid.csv', index=False)
        df_test.to_csv('./data/preprocessed/test.csv', index=False)
        self.stdout.write(self.style.SUCCESS('PREPROCESSING FINISHED.'))


