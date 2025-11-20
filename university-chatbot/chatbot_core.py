import sqlite3
import re
from datetime import datetime
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
import warnings
import google.generativeai as genai

# --- CONFIGURATION ---
GOOGLE_API_KEY = "YOUR_API_KEY_HERE"
DB_PATH = 'university.db'

genai.configure(api_key=GOOGLE_API_KEY)
try:
    model = genai.GenerativeModel('gemini-pro')
except:
    model = None

# Import training data
from dataset import TRAINING_QUERIES

warnings.filterwarnings('ignore')

def download_nltk_data():
    resources = ['punkt', 'wordnet', 'stopwords']
    for res in resources:
        try:
            nltk.data.find(f'tokenizers/{res}') if res == 'punkt' else nltk.data.find(f'corpora/{res}')
        except LookupError:
            nltk.download(res, quiet=True)
download_nltk_data()

# --- DB HELPER ---
def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

class NLPProcessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

    def preprocess(self, text):
        tokens = word_tokenize(text.lower())
        tokens = [self.lemmatizer.lemmatize(t) for t in tokens if t.isalnum() and t not in self.stop_words]
        return " ".join(tokens)

class IntentClassifier:
    def __init__(self, processor):
        self.processor = processor
        self.vectorizer = TfidfVectorizer()
        self.classifier = SVC(kernel='linear', probability=True)
        self.train()

    def train(self):
        queries, labels = zip(*TRAINING_QUERIES)
        processed_queries = [self.processor.preprocess(q) for q in queries]
        X_vectors = self.vectorizer.fit_transform(processed_queries)
        self.classifier.fit(X_vectors, labels)

    def predict(self, text):
        processed = self.processor.preprocess(text)
        vector = self.vectorizer.transform([processed])
        intent = self.classifier.predict(vector)[0]
        probs = self.classifier.predict_proba(vector)
        return intent, np.max(probs)

class EntityMatcher:
    def __init__(self, processor):
        self.processor = processor
        self.vectorizer = TfidfVectorizer()

    def find_best_match_sql(self, user_query, table, columns):
        conn = get_db_connection()
        rows = conn.execute(f"SELECT * FROM {table}").fetchall()
        conn.close()

        if not rows: return None, 0
        
        search_candidates = [" ".join([str(row[col]) for col in columns if row[col]]) for row in rows]
        
        processed_query = self.processor.preprocess(user_query)
        processed_candidates = [self.processor.preprocess(c) for c in search_candidates]
        
        all_text = processed_candidates + [processed_query]
        tfidf_matrix = self.vectorizer.fit_transform(all_text)
        
        query_vec = tfidf_matrix[-1]
        candidate_vecs = tfidf_matrix[:-1]
        
        cosine_scores = cosine_similarity(query_vec, candidate_vecs).flatten()
        best_idx = np.argmax(cosine_scores)
        return rows[best_idx], cosine_scores[best_idx]

class UniversityChatbot:
    def __init__(self):
        self.processor = NLPProcessor()
        self.intent_classifier = IntentClassifier(self.processor)
        self.entity_matcher = EntityMatcher(self.processor)

    def generate_ai_response(self, user_query, context_data):
        try:
            if "YOUR_API_KEY" in GOOGLE_API_KEY or not model: raise Exception()
            prompt = f"Answer based on this data: {context_data}. User asked: {user_query}"
            response = model.generate_content(prompt)
            return response.text
        except:
            return f"Here is the information I found:\n\n{context_data}"

    # --- DEADLINE LOGIC ---
    def add_deadline_logic(self, message):
        match = re.search(r'add deadline (.+?) by (\d{4}-\d{2}-\d{2})', message, re.IGNORECASE)
        if match:
            title = match.group(1)
            date = match.group(2)
            conn = get_db_connection()
            conn.execute("INSERT INTO deadlines (title, due_date, status) VALUES (?, ?, 'pending')", (title, date))
            conn.commit()
            conn.close()
            return f"✅ Added: **{title}** (Due: {date})"
        return "To add a task, type: **'Add deadline [Name] by YYYY-MM-DD'**"

    def mark_deadline_complete(self, message):
        match = re.search(r'mark (.+?) as (done|completed)', message, re.IGNORECASE)
        if match:
            title_query = match.group(1).strip()
            conn = get_db_connection()
            rows = conn.execute("SELECT * FROM deadlines").fetchall()
            
            target_id = None
            target_title = ""
            for r in rows:
                if title_query.lower() in r['title'].lower():
                    target_id = r['id']
                    target_title = r['title']
                    break
            
            if target_id:
                conn.execute("UPDATE deadlines SET status = 'completed' WHERE id = ?", (target_id,))
                conn.commit()
                conn.close()
                return f"🎉 Marked **{target_title}** as completed!"
            else:
                conn.close()
                return f"Could not find task '{title_query}'."
        return "Type **'Mark [Name] as done'** to complete a task."

    def get_deadline_status(self, filter_type):
        conn = get_db_connection()
        today = datetime.now().strftime('%Y-%m-%d')
        
        if filter_type == 'upcoming':
            rows = conn.execute("SELECT * FROM deadlines WHERE due_date >= ? ORDER BY due_date ASC", (today,)).fetchall()
            header = "📅 **Upcoming Deadlines:**"
        elif filter_type == 'passed':
            rows = conn.execute("SELECT * FROM deadlines WHERE due_date < ? ORDER BY due_date DESC", (today,)).fetchall()
            header = "⚠️ **Past/History:**"
        
        conn.close()
        
        if not rows: return f"{header}\nNo tasks found."
            
        text = f"{header}\n"
        for r in rows:
            marker = "✅" if r['status'] == 'completed' else "⏳"
            text += f"{marker} **{r['title']}**: {r['due_date']}\n"
        return text

    def get_response(self, message):
        intent, confidence = self.intent_classifier.predict(message)
        
        if "add deadline" in message.lower():
            return {'response': self.add_deadline_logic(message), 'data': None}
        if "mark" in message.lower() and ("done" in message.lower() or "completed" in message.lower()):
            return {'response': self.mark_deadline_complete(message), 'data': None}

        if intent == 'greeting':
            return {'response': "Hello! I am BU Buddy. I can help with Syllabus, Professors, Locations, and Deadlines.", 'data': None}

        context_info = ""
        pdf_url = None

        if intent == 'syllabus':
            row, score = self.entity_matcher.find_best_match_sql(message, 'courses', ['code', 'name'])
            if score > 0.15:
                conn = get_db_connection()
                syl = conn.execute("SELECT * FROM syllabus WHERE course_code=?", (row['code'],)).fetchone()
                conn.close()
                if syl:
                    return {'response': f"Here is the Syllabus PDF for **{row['name']}**.", 'data': {'pdf_url': syl['pdf_url']}}
                context_info = "Course found, but syllabus PDF is missing."

        elif intent == 'pyq':
            row, score = self.entity_matcher.find_best_match_sql(message, 'courses', ['code', 'name', 'dept'])
            if score > 0.15:
                conn = get_db_connection()
                pyq = conn.execute("SELECT * FROM pyqs WHERE course_code=?", (row['code'],)).fetchone()
                conn.close()
                if pyq:
                     return {'response': f"Here is the Past Year Paper for **{row['name']}**.", 'data': {'pdf_url': pyq['pdf_url']}}
                context_info = f"No PYQs found for {row['name']}."

        elif intent == 'professor':
            if "list" in message.lower():
                conn = get_db_connection()
                rows = conn.execute("SELECT name, dept FROM professors").fetchall()
                conn.close()
                context_info = "Professors:\n" + "\n".join([f"- {r['name']} ({r['dept']})" for r in rows])
            else:
                row, score = self.entity_matcher.find_best_match_sql(message, 'professors', ['name', 'specialization', 'dept'])
                if score > 0.15:
                    context_info = f"👤 **{row['name']}**\n🏢 {row['office']}\n📧 {row['email']}\nSpec: {row['specialization']}"

        elif intent == 'location':
            # 1. Try Location Table
            row, score = self.entity_matcher.find_best_match_sql(message, 'locations', ['name'])
            if score > 0.2:
                 context_info = f"📍 **{row['name']}** is in **{row['building']}** (Floor {row['floor']})."
            else:
                # 2. Fallback to Professor (Smart Switching)
                p_row, p_score = self.entity_matcher.find_best_match_sql(message, 'professors', ['name'])
                if p_score > 0.15:
                    context_info = f"👤 **{p_row['name']}** sits in **{p_row['office']}**."

        elif intent == 'deadline':
            return {'response': self.get_deadline_status('upcoming'), 'data': None}
        
        elif intent == 'deadline_history':
            return {'response': self.get_deadline_status('passed'), 'data': None}

        if context_info:
            return {'response': self.generate_ai_response(message, context_info), 'data': None}

        return {'response': "I couldn't find that info. Please check the spelling.", 'data': None}