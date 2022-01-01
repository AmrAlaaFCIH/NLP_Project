# Question Answering For Arabic Data

## Getting Started
**Get a deeper understanding of our main models:**
- Scrap The Data from  https://www.arab-books.com
- Cleaning and Autocorrect The Data
- Making a Pipeline to classify The Data using LogisticRegression and RandomForestClassifier
- Build a simple website containing Question Answering System

## ScrapingThe Data:
In The DataScraping model, we scrap an arabic website for books and get some info about each book:
- الكتاب 
- مؤلف الكتاب
- قسم الكتاب
- عدد الصفحات
- دار النشر
- الملخص
- عنوان كل كتاب 
- حجم

```
import requests
from bs4 import BeautifulSoup
import csv
import re
```

```
counter = 1
with open("Extracted_data.csv", newline='', mode='w', encoding="utf-8-sig") as output_file:
    writer = csv.DictWriter(output_file,
                            ["Author", "Book_Name", "Book_Category", "Book_url", "Publishing_house", "Book_size",
                             "Book_pages", "Quotation"])
    writer.writeheader()
    while True:
        request = requests.get("https://www.arab-books.com//page/{}".format(counter))
        
        #end loop 
        if request.status_code == 404:
            break

        soup = BeautifulSoup(request.text)

        if not soup.select("#posts-container > li"):
            break

        for book in soup.select("#posts-container > li"):
            book_name = None
            book_url = None
            if len(book.select("a[aria-label]")) > 0:
                book_url = book.select("a[aria-label]")[0]['href']
                book_name = book.select("a[aria-label]")[0]['aria-label']
            book_category = None
            book_house = None
            book_quote = None
            book_size = None
            book_pages = None
            if book_url is not None:
                book_soup = None
                try:
                    book_soup = BeautifulSoup(requests.get(book_url).text)
                except Exception:
                    book_soup = None
                    
                if book_soup is not None:      
                    try:
                        book_category = '-'.join(
                            [x.text for x in book_soup.select("div.entry-header > span.post-cat-wrap > a")])
                    except Exception:
                        book_category = None
    
                    try:
                        book_house = re.search(r'<li><strong>دار النشر:</strong>(.*?)</li>',
                                               str(book_soup.select("div.book-info")[0])).group(1).strip()
                    except Exception:
                        book_house = None
    
                    try:
                        book_quote = ' '.join(re.sub(r'[^\u0621-\u064A\u0660-\u0669]', ' ',
                                                     book_soup.select("div.peotry_one_line")[0].text).split())
                    except Exception:
                        book_quote = None
    
                    try:
                        book_size = re.search(r'<li><strong>حجم الكتاب:</strong>(.*?)</li>',
                                              str(book_soup.select("div.book-info")[0])).group(1).strip()
                    except Exception:
                        book_size = None
    
                    try:
                        book_pages = re.search(r'<li><strong>عدد الصّفحات:</strong>(.*?)</li>',
                                               str(book_soup.select("div.book-info")[0])).group(1).strip()
                    except Exception:
                        book_pages = None

            author = None
            if len(book.select("div.book-writer > a")) > 0:
                author = book.select("div.book-writer > a")[0].text
            writer.writerow({
                "Author": author,
                "Book_Name": book_name,
                "Book_Category": book_category,
                "Book_url": book_url,
                "Publishing_house": book_house,
                "Book_size": book_size,
                "Book_pages": book_pages,
                "Quotation": book_quote
            })
        print(f"Page {counter} finished.....")
        counter += 1
```
## Cleaning The Data
First, we check for the Null values
```
import string
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
pd.options.mode.chained_assignment = None
df=pd.read_csv('Extracted_data.csv')
```

```
quotation_null=df["Quotation"].isna().sum() / df.shape[0] *100
print(f"Quotation Column is {quotation_null}% empty")
# we will ignore it for now
#check other null data
df[df["Book_pages"].isna() | df["Book_size"].isna() | df["Book_url"].isna()]
df.isna().sum()
```

we will use the books with missing category later to test our model, so we will leave it for now and start cleaning text data. next, we will clean each column from extra text:
```
df["Book_size"]=df["Book_size"].str.replace(r'[^\d.]','',regex=True)
df.rename(columns={'Book_size':'Book_size_MB'},inplace=True)
df.head()
```
```
df["Book_pages"]=df["Book_pages"].str.replace(r'[^\d]','',regex=True)
```
```
book_name=[]
for word in df['Book_Name']:
    phase1=re.sub(r'[Pp][Dd][Ff]|كتاب|[!":,.؟]','',word)
    phase2=phase1.split('للكاتب')[0]
    phase3=re.sub(r'\s{2,}',' ',phase2)
    phase4=phase3.strip()
    book_name.append(phase4)

df['Book_Name']=book_name
df=df[~df['Book_Name'].str.match(u'[^\u0621-\u064A\u0660-\u0669a-zA-Z0-9]')]
```
```
df["Publishing_house"]=df["Publishing_house"].str.replace(u'[^\u0621-\u064A\u0660-\u0669\s]','',regex=True)
```
```
df.replace(r'^\s*$', np.NaN, regex=True,inplace=True)
```
## Autocorrect The Data:
here we correct the data from any missing letters based on an Arabic dictionary called "Khaleej".
```
from sklearn.datasets import load_files
from collections import Counter
from nltk.tokenize import word_tokenize
import pandas as pd
import re
```
```
class AutoCorrect:
    def from_file(self,file):
        vocablist=[]
        with open(file,encoding='utf-8-sig',mode='r') as file:
            lines=file.readlines()
            for line in lines:
                vocablist +=[word for word in re.findall(r'\w+',line) if len(word)>1]
        vocab_counter=Counter(vocablist)
        totalwords=sum(vocab_counter.values())
        self.vocab=set(vocablist)
        self.word_prob={word:vocab_counter[word]/totalwords for word in vocab_counter.keys()}

    def from_dir(self,dir):
        dataset=load_files(dir,encoding='utf-8-sig')
        vocablist=[]
        for doc in dataset['data']:
            vocablist +=[word for word in re.findall(r'\w+',doc) if len(word)>1]
        vocab_counter=Counter(vocablist)
        totalwords=sum(vocab_counter.values())
        self.vocab=set(vocablist)
        self.word_prob={word:vocab_counter[word]/totalwords for word in vocab_counter.keys()}


    def _level_one_edits(self,word):
        letters=u'ابتةثجحخدذرزسشصضطظعغفقكلمنهويءآأؤإئ'
        splits=[(word[:part],word[part:]) for part in range(0,len(word))]
        deletes=[part1+part2[1:] for part1,part2 in splits if part2]
        swaps=[part1+part2[0]+part2[1]+part2[2:] for part1,part2 in splits if len(part2) > 1]
        replaces=[part1+c+part2[1:] for part1,part2 in splits if part2 for c in letters]
        inserts=[part1+c+part2 for part1,part2 in splits for c in letters]
        return set(deletes+swaps+replaces+inserts)

    def _level_two_edits(self,word):
        return set(e2 for e1 in self._level_one_edits(word) for e2 in self._level_one_edits(e1))

    def check(self,word):
        if word in self.vocab:
            return True
        candidates =list(self._level_one_edits(word))+list(self._level_two_edits(word))+[word]
        valid_candidates = [w for w in candidates if w in self.vocab]
        if len(valid_candidates) > 0 :
            return sorted([(c, self.word_prob[c]) for c in valid_candidates], key=lambda tup: tup[1], reverse=True)
        else:
            return False

    def correct(self,word):
        if word in self.vocab:
            return word
        candidates =list(self._level_one_edits(word))+list(self._level_two_edits(word))+[word]
        valid_candidates = [w for w in candidates if w in self.vocab]
        if len(valid_candidates) > 0 :
            return sorted([(c, self.word_prob[c]) for c in valid_candidates], key=lambda tup: tup[1], reverse=True)[0][0]
        else:
            return word
```
```
autoCorrect=AutoCorrect()
autoCorrect.from_dir('Khaleej-2004')
#Auto Correct data
book_names=[]
book_Quotation=[]
for record in df["Book_Name"]:
    record=re.sub("\d","",record)
    words=word_tokenize(record)
    words_new=[]
    for word in words:
        if len(word) < 2:
            words_new.append(autoCorrect.correct(word))
        else:
            words_new.append(word)
    book_names.append(' '.join(words_new))

for record in df[~(df["Quotation"].isna())]["Quotation"]:
    record=re.sub("\d","",record)
    words=word_tokenize(record)
    words_new=[]
    for word in words:
        if len(word) < 2:
            words_new.append(autoCorrect.correct(word))
        else:
            words_new.append(word)
    book_Quotation.append(' '.join(words_new))
```
```
df["Book_Name"]=book_names
df[~(df["Quotation"].isna())]["Quotation"]=book_Quotation
```
## Making a Pipeline to classify The Data:
```
#split data to two models one with Quotation and with not
df1=df[df["Quotation"].isna()]
df2=df[~(df["Quotation"].isna())]
```
prepare the training for model 1 that use LogisticRegression with Quotation data.
```
#prepare the training for model 1
x1_test=df1[df1["Book_Category"].isna()]["Book_Name"].tolist()
x1_train=df1[~(df1["Book_Category"].isna())]["Book_Name"].tolist()
y1_train=df1[~(df1["Book_Category"].isna())]["Book_Category"].tolist()
```
```
pipe=Pipeline([("Vector",TfidfVectorizer(encoding='utf-8-sig',lowercase=False)),("Model",LogisticRegression(max_iter=2000))])
pipe.fit(x1_train,y1_train)
```
Prepare the training for model 2 that use LogisticRegression with out Quotation data.
```
#prepare the training for model 2
x2_test=df2[df2["Book_Category"].isna()]["Quotation"].tolist()
x2_train=df2[~(df2["Book_Category"].isna())]["Quotation"].tolist()
y2_train=df2[~(df2["Book_Category"].isna())]["Book_Category"].tolist()
```
```
pipe=Pipeline([("Vector",TfidfVectorizer(encoding='utf-8-sig',lowercase=False)),("Model",LogisticRegression(max_iter=2000))])
pipe.fit(x2_train,y2_train)
```
## Build a Question Answering System:
here we make questions from the data then train a NearestNeighbors model wiht questions and there anwsers.
```
import joblib
import pandas as pd 
from  yake import KeywordExtractor
import  numpy as np


df=pd.read_csv(r"E:\#Universty_Resources\Level 4\Selected-3\Project\Cleaned_data.csv")
df.drop(columns="Quotation",inplace=True)
df.dropna(inplace=True)
def process_context(data,question,answer):
  data["Question"].append(question)
  data['Answer'].append(answer)


#dectionary to store data with answers
data={
    'Question':[],
    'Answer':[]
}
```
Create questions about the dataset
```
for i,row in df[["Book_Name","Author","Book_Category","Publishing_house","Book_size_MB","Book_pages"]].iterrows() :
  question1=f"من مؤلف كتاب {row['Book_Name']} ؟"
  question2=f"ما هى دار النشر المسئولة عن كتاب {row['Book_Name']} ؟"
  question3=f"ما عدد صفحات كتاب {row['Book_Name']} ؟"
  process_context(data,question1,row['Author'])
  process_context(data,question2,row['Publishing_house'])
  process_context(data,question3,int(row['Book_pages']))

  
answers=pd.DataFrame(data)
answers.head()
```
Model learn
```
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
vectorizer=TfidfVectorizer(lowercase=False)
vectorizer.fit(answers["Question"])
model=NearestNeighbors(n_neighbors=1)
model.fit(vectorizer.transform(answers["Question"]))

def predict(x):
  return answers.iloc[model.kneighbors(vectorizer.transform([x]))[1][0][0],1] 
```
Extract important keywords from dataset using yake library
```
from yake import KeywordExtractor
extractor=KeywordExtractor(lan="ar",n=3,top=2)
df=pd.read_csv(r"E:\#Universty_Resources\Level 4\Selected-3\Project\Cleaned_data.csv")
text=df[~(df["Quotation"].isna())]["Quotation"].to_list()
text[:2]

keywords=[]
for line in text:
  words=extractor.extract_keywords(line)
  keywords.extend(words)
keywords= list(set(np.array(keywords)[:,0]))
```
