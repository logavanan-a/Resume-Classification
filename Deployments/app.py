import streamlit as st
import pickle as pk
import docx2txt
import re
import pdfplumber
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas as pd

model = pk.load(open(r'model_xgb.pkl', 'rb'))
Vectorizer = pk.load(open(r'vector.pkl', 'rb'))
stop_words = set(stopwords.words('english'))

def preprocess(sentence):
    sentence = str(sentence)
    sentence = sentence.lower()
    sentence = sentence.replace('{html}',"") 
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', sentence)
    rem_url = re.sub(r'http\S+', '',cleantext)
    rem_num = re.sub('[0-9]+', '', rem_url)
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(rem_num)  
    filtered_words = [w for w in tokens if len(w) > 2 if not w in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    lemma_words = [lemmatizer.lemmatize(w) for w in filtered_words]
    return " ".join(lemma_words) 

def extract_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            txt = page.extract_text()
            if txt:
                text += txt + " "
    return text

def extract_docx(file):
    return docx2txt.process(file)

st.title('RESUME CLASSIFICATION')
st.markdown('<style>h1{color: Purple;}</style>', unsafe_allow_html=True)
st.subheader('Welcome to Resume Classification App')
uploaded_files = st.file_uploader('Upload Your Resumes', type= ['docx','pdf'],accept_multiple_files=True)

select = ['PeopleSoft','SQL Developer','React JS Developer','Workday']
st.subheader('Select as per Requirement')
option = st.selectbox('Fields',select)


if st.button("Predict for All Files"):
    if not uploaded_files:
        st.warning("Please upload at least one file!")
    else:
        results = []
        for file in uploaded_files:
            file_name = file.name
            ext = file_name.split(".")[-1].lower()
            if ext == "pdf":
                raw_text = extract_pdf(file)
            elif ext == "docx":
                raw_text = extract_docx(file)
            else:
                st.error(f"Unsupported file type: {file_name}")
                continue

            cleaned = preprocess(raw_text)
            vectorized = Vectorizer.transform([cleaned])

            prediction = model.predict(vectorized)[0]
        
            results.append({
                "File Name": file_name,
                "Predicted Category": prediction,
            })

        if len(results) > 0:
            df_results = pd.DataFrame(results)
            st.table(df_results)
            st.download_button(
                label="Download Skills CSV",
                data=df_results.to_csv(index=False),
                file_name="extracted_skills.csv",
                mime="text/csv"
            )
            # category = encoder.inverse_transform([prediction])[0]

