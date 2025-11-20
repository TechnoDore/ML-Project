import sqlite3
import json
from datetime import datetime, timedelta
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
# REPLACE WITH YOUR ACTUAL KEY
GOOGLE_API_KEY = "YOUR_API_KEY_HERE" 
DB_PATH = 'university.db'

genai.configure(api_key=GOOGLE_API_KEY)
try:
    model = genai.GenerativeModel('gemini-pro')
except:
    model = None

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

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

class NLPProcessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

    def preprocess(self, text):
        text = text.lower()
        tokens = word_tokenize(text)
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

    # 1. STANDARD SEARCH (TF-IDF)
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
        
        # Higher threshold for TF-IDF to avoid bad guesses
        if cosine_scores[best_idx] > 0.2:
            return rows[best_idx], cosine_scores[best_idx]
        
        # --- SEMANTIC AI FALLBACK ---
        # If math fails (e.g., "ML" vs "Artificial Intelligence"), ask the AI
        return self.find_semantic_match_ai(user_query, rows, columns)

    # 2. SEMANTIC SEARCH (AI-POWERED)
    def find_semantic_match_ai(self, user_query, rows, columns):
        if not model: return None, 0
        
        # Create a readable list for the AI
        options = [f"{i}: " + " ".join([str(row[col]) for col in columns if row[col]]) for i, row in enumerate(rows)]
        options_str = "\n".join(options)
        
        prompt = (
            f"User is searching for: '{user_query}'.\n"
            f"Match it to one of these database entries:\n{options_str}\n"
            f"Return ONLY the index number of the best match. If unsure, return -1."
        )
        
        try:
            response = model.generate_content(prompt)
            index = int(response.text.strip())
            if index != -1 and 0 <= index < len(rows):
                return rows[index], 0.9 # High confidence since AI picked it
        except:
            pass
            
        return None, 0

    def global_search(self, user_query):
        tables_to_search = [
            ('professors', ['name', 'specialization', 'dept', 'office']),
            ('courses', ['name', 'code', 'dept']),
            ('locations', ['name', 'building'])
        ]
        
        best_match = None
        best_score = 0
        best_table = ""

        for table, columns in tables_to_search:
            row, score = self.find_best_match_sql(user_query, table, columns)
            if score > best_score:
                best_score = score
                best_match = row
                best_table = table

        if best_score > 0.2:
            return best_match, best_table
        return None, None

class UniversityChatbot:
    def __init__(self):
        self.processor = NLPProcessor()
        self.intent_classifier = IntentClassifier(self.processor)
        self.entity_matcher = EntityMatcher(self.processor)

    # --- AI HELPER: DATA EXTRACTION ---
    def extract_data_with_ai(self, message, extraction_type):
        if not model: return None
        
        today_str = datetime.now().strftime('%Y-%m-%d')
        
        if extraction_type == "deadline":
            prompt = (
                f"Today is {today_str}. Extract task title and due date from: '{message}'. "
                f"Convert relative dates like 'next Friday' or 'tomorrow' to YYYY-MM-DD format. "
                f"Return ONLY a JSON object: {{'title': 'Task Name', 'date': 'YYYY-MM-DD'}}."
            )
        elif extraction_type == "task_name":
            prompt = (
                f"Extract the task name to be marked as done from: '{message}'. "
                f"Return ONLY a JSON object: {{'title': 'Task Name'}}."
            )
        
        try:
            response = model.generate_content(prompt)
            clean_text = response.text.replace('```json', '').replace('```', '').strip()
            return json.loads(clean_text)
        except:
            return None

    def generate_ai_response(self, user_query, context_data):
        try:
            if "YOUR_API_KEY" in GOOGLE_API_KEY or not model: 
                raise Exception("Missing API Key")
            
            prompt = (
                f"You are 'BU Buddy'. Answer based on this data: {context_data}. "
                f"Be concise. If asked for specific details (like office), only give that. "
                f"Fix spelling. User Question: {user_query}"
            )
            response = model.generate_content(prompt)
            return response.text
        except:
            return f"Here is the information I found:\n\n{context_data}"

    # --- LOGIC HANDLERS ---
    def add_deadline_logic(self, message):
        data = self.extract_data_with_ai(message, "deadline")
        if data and 'title' in data and 'date' in data:
            conn = get_db_connection()
            conn.execute("INSERT INTO deadlines (title, due_date, status) VALUES (?, ?, 'pending')", (data['title'], data['date']))
            conn.commit()
            conn.close()
            return f"✅ Added: **{data['title']}** (Due: {data['date']})"
        return "I couldn't understand the task details. Try: 'Add deadline [Name] by [Date]'."

    def mark_deadline_complete(self, message):
        data = self.extract_data_with_ai(message, "task_name")
        if data and 'title' in data:
            query_title = data['title']
            conn = get_db_connection()
            rows = conn.execute("SELECT * FROM deadlines").fetchall()
            
            # Use Semantic Search for this too
            matcher = EntityMatcher(NLPProcessor())
            best_row, score = matcher.find_semantic_match_ai(query_title, rows, ['title'])
            
            if best_row:
                conn.execute("UPDATE deadlines SET status = 'completed' WHERE id = ?", (best_row['id'],))
                conn.commit()
                conn.close()
                return f"🎉 Marked **{best_row['title']}** as completed!"
            
            conn.close()
            return f"Could not find a task matching '{query_title}'."
        return "Which task would you like to mark as done?"

    def get_deadline_status(self, filter_type):
        conn = get_db_connection()
        today = datetime.now().strftime('%Y-%m-%d')
        query = "SELECT * FROM deadlines WHERE due_date >= ? ORDER BY due_date ASC" if filter_type == 'upcoming' else "SELECT * FROM deadlines WHERE due_date < ? ORDER BY due_date DESC"
        header = "📅 **Upcoming Deadlines:**" if filter_type == 'upcoming' else "⚠️ **Past/History:**"
        rows = conn.execute(query, (today,)).fetchall()
        conn.close()
        if not rows: return f"{header}\nNo tasks found."
        text = f"{header}\n"
        for r in rows:
            marker = "✅" if r['status'] == 'completed' else "⏳"
            text += f"{marker} **{r['title']}**: {r['due_date']}\n"
        return text

    # --- MAIN RESPONSE FLOW ---
    def get_response(self, message):
        intent, confidence = self.intent_classifier.predict(message)
        
        if intent == 'add_deadline_intent':
            return {'response': self.add_deadline_logic(message), 'data': None}
        elif intent == 'mark_deadline_intent':
            return {'response': self.mark_deadline_complete(message), 'data': None}
        elif intent == 'deadline':
            return {'response': self.get_deadline_status('upcoming'), 'data': None}
        elif intent == 'deadline_history':
            return {'response': self.get_deadline_status('passed'), 'data': None}
        elif intent == 'greeting':
            return {'response': "Hello! I am BU Buddy. I can help with Syllabus, Professors, Locations, and Deadlines.", 'data': None}

        context_info = None
        
        if intent == 'syllabus':
            row, score = self.entity_matcher.find_best_match_sql(message, 'courses', ['code', 'name'])
            if row:
                conn = get_db_connection()
                syl = conn.execute("SELECT * FROM syllabus WHERE course_code=?", (row['code'],)).fetchone()
                conn.close()
                if syl: return {'response': f"Here is the Syllabus PDF for **{row['name']}**.", 'data': {'pdf_url': syl['pdf_url']}}
                context_info = f"Course {row['name']} found, but PDF is missing."
        
        elif intent == 'pyq':
            row, score = self.entity_matcher.find_best_match_sql(message, 'courses', ['code', 'name'])
            if row:
                conn = get_db_connection()
                pyq = conn.execute("SELECT * FROM pyqs WHERE course_code=?", (row['code'],)).fetchone()
                conn.close()
                if pyq: return {'response': f"Here is the PYQ for **{row['name']}**.", 'data': {'pdf_url': pyq['pdf_url']}}

        elif intent == 'professor':
            row, score = self.entity_matcher.find_best_match_sql(message, 'professors', ['name', 'specialization', 'dept'])
            if row: 
                context_info = f"Name: {row['name']}, Office: {row['office']}, Email: {row['email']}, Spec: {row['specialization']}"

        elif intent == 'location':
            row, score = self.entity_matcher.find_best_match_sql(message, 'locations', ['name', 'building'])
            if row: context_info = f"Location: {row['name']} is in {row['building']} (Floor {row['floor']})."

        if not context_info:
            row, table_name = self.entity_matcher.global_search(message)
            if row:
                if table_name == 'professors':
                    context_info = f"Found Professor: {row['name']}, Office: {row['office']}, Email: {row['email']}"
                elif table_name == 'courses':
                     context_info = f"Found Course: {row['name']} ({row['code']}). Credits: {row['credits']}."
                elif table_name == 'locations':
                    context_info = f"Found Location: {row['name']} in {row['building']}."

        if context_info:
            return {'response': self.generate_ai_response(message, context_info), 'data': None}

        return {'response': "I couldn't find that info. Please try rephrasing.", 'data': None}