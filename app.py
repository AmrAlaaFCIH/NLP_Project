import joblib
import pandas as pd 
from  yake import KeywordExtractor
import  numpy as np


df=pd.read_csv("Cleaned_data.csv")
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

#create questions about dataset 
for i,row in df[["Book_Name","Author","Book_Category","Publishing_house","Book_size_MB","Book_pages"]].iterrows() :
  question1=f"من مؤلف كتاب {row['Book_Name']} ؟"
  question2=f"ما هى دار النشر المسئولة عن كتاب {row['Book_Name']} ؟"
  question3=f"ما عدد صفحات كتاب {row['Book_Name']} ؟"
  process_context(data,question1,row['Author'])
  process_context(data,question2,row['Publishing_house'])
  process_context(data,question3,int(row['Book_pages']))

  
answers=pd.DataFrame(data)
answers.head()

# model learn 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
vectorizer=TfidfVectorizer(lowercase=False)
vectorizer.fit(answers["Question"])
model=NearestNeighbors(n_neighbors=1)
model.fit(vectorizer.transform(answers["Question"]))

def predict(x):
  return answers.iloc[model.kneighbors(vectorizer.transform([x]))[1][0][0],1] 


#extract keywords from dataset
from yake import KeywordExtractor
extractor=KeywordExtractor(lan="ar",n=3,top=2)
df=pd.read_csv("Cleaned_data.csv")
text=df[~(df["Quotation"].isna())]["Quotation"].to_list()
text[:2]

keywords=[]
for line in text:
  words=extractor.extract_keywords(line)
  keywords.extend(words)
keywords= list(set(np.array(keywords)[:,0]))




from flask import Flask, render_template, jsonify, request
app = Flask(__name__)
app.static_folder = 'static'


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/answer-question', methods=['POST'])
def analyzer():
    
    #get question from gui 
    data = request.get_json()
    question = data.get('question')
 
    result = predict(question)
    print("code is running")
    return jsonify({"answer": result, "keywords" : keywords })


if __name__ == '__main__':
    app.run(debug=True)
