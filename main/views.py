import requests
from django.http import JsonResponse
from django.shortcuts import render
from lxml.html import fromstring

# Create your views here.
from main.models import dataset
from utils import model


def lemmatization(text):
    from nltk.stem import WordNetLemmatizer
    from nltk import pos_tag, word_tokenize
    import nltk
    nltk.download('wordnet')
    nltk.download("stopwords")
    nltk.download('averaged_perceptron_tagger')
    nltk.download('punkt')
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


def index(request):
    msg = ""
    if request.method == "POST":
        author = request.POST['author']
        label = request.POST['label']
        title = request.POST['title']
        text = request.POST['text']
        dataset(author=author, label=label, title=title, text=text).save()
        msg = "Thank you for helping us !"
    return render(request, "index.html", {"msg": msg})


def checkUrl(request):
    print('getting title')
    url = request.GET.get('url')
    r = requests.get(url)
    print('parsing page')
    tree = fromstring(r.content)
    title = tree.findtext('.//title')
    title = lemmatization(title)
    prediction = round(float(model.predict(title)), 2)
    if prediction < 0.50:
        msg = "This article seems REAL !"
    else:
        if prediction > 0.75:
            msg = "This article is  Fake !"
        else:
            msg = "This article is probably Fake !"

    response = {'percent': prediction, 'msg': msg}

    return JsonResponse(response)


def checkText(request):
    print('getting title')
    article = request.GET.get('article')
    title = request.GET.get('title')
    text = lemmatization(title + " " + article)
    if float(model.predict(text)) < 0.50:
        msg = "This article seems REAL !"
    else:
        if float(model.predict(text)) > 0.75:
            msg = "This article is  Fake !"
        else:
            msg = "This article is probably Fake !"

    response = {'percent': round(model.predict(text), 2), 'msg': msg}

    return JsonResponse(response)
