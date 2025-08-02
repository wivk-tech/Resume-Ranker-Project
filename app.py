import os
import re
import PyPDF2
import spacy
from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ======== CONFIGURATION ========
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Create Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ======== FUNCTIONS ========
def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    text = ""
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted
    return text

def preprocess_text(text):
    """Lowercase, remove special chars, lemmatize, remove stopwords."""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop]
    return " ".join(tokens)

def rank_resumes(resume_texts, job_description):
    """Rank resumes based on similarity to job description."""
    all_texts = [job_description] + resume_texts
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(all_texts)

    scores = cosine_similarity(vectors[0:1], vectors[1:]).flatten()
    ranking = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)

    return [(idx, score) for idx, score in ranking]

# ======== ROUTES ========
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get Job Description
        jd_text = request.form['job_description']
        processed_jd = preprocess_text(jd_text)

        # Process Resumes
        resume_texts = []
        resume_files = []

        for file in request.files.getlist('resumes'):
            if file and file.filename.endswith('.pdf'):
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                file.save(filepath)

                raw_text = extract_text_from_pdf(filepath)
                processed_text = preprocess_text(raw_text)

                resume_texts.append(processed_text)
                resume_files.append(file.filename)

        # Rank resumes
        results = rank_resumes(resume_texts, processed_jd)
        ranked_data = [(resume_files[idx], score) for idx, score in results]

        return render_template('results.html', ranked_data=ranked_data)

    return render_template('index.html')

# ======== ENTRY POINT ========
if __name__ == "__main__":
    # Run with: python app.py
    app.run(debug=True)
