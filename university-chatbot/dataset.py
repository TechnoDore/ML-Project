# dataset.py
# Expanded training sentences with ACRONYMS to teach the ML model.

TRAINING_QUERIES = [
    # --- Syllabus & Course Content ---
    ("syllabus for Engineering Calculus", "syllabus"), 
    ("what is covered in CSET243", "syllabus"),
    ("course content for Information Management Systems", "syllabus"), 
    ("topics in High Performance Computing", "syllabus"),
    ("what to read for Exam", "syllabus"),
    ("syllabus for ML", "syllabus"),
    ("topics in DS", "syllabus"),
    ("web sec syllabus", "syllabus"),
    ("AI course content", "syllabus"),

    # --- Professor & Contact Info ---
    ("who teaches Data Structures", "professor"), 
    ("contact details for the UI/UX professor", "professor"), 
    ("where does Dr. Vikram sit", "professor"), 
    ("professor's room number", "professor"),
    ("who teaches ML", "professor"),
    ("prof for DS", "professor"),
    ("faculty for AI", "professor"),
    ("Web Security teacher", "professor"),
    ("Priya ma'am office", "professor"),
    ("Vikram sir cabin", "professor"),

    # --- PYQ (Past Year Questions) ---
    ("past year questions for Probability and Statistics", "pyq"), 
    ("previous exam papers for CSET201", "pyq"),
    ("question paper for physics", "pyq"),
    ("PYQ for ML", "pyq"),
    ("DS previous papers", "pyq"),
    ("last year paper AI", "pyq"),

    # --- Deadlines (Reading) ---
    ("when is the assignment due", "deadline"), 
    ("upcoming deadlines", "deadline"), 
    ("is there any work pending", "deadline"),
    ("show passed deadlines", "deadline_history"),
    ("history of tasks", "deadline_history"),

    # --- Deadlines (Writing) ---
    ("add deadline", "add_deadline_intent"),
    ("I want to add a task", "add_deadline_intent"),
    ("remind me to submit project", "add_deadline_intent"),
    ("submit ML assignment next friday", "add_deadline_intent"),
    ("add web sec project by monday", "add_deadline_intent"),
    ("mark task as done", "mark_deadline_intent"),
    ("I finished the Web Sec Project", "mark_deadline_intent"),
    ("ML assignment completed", "mark_deadline_intent"),
    ("mark DS homework as done", "mark_deadline_intent"),

    # --- Location ---
    ("where is the library", "location"), 
    ("how to get to the computer lab", "location"),
    ("location of cafeteria", "location"),
    ("lab location", "location"),
    ("where is the canteen", "location"),

    # --- Greeting ---
    ("hi", "greeting"), 
    ("hello", "greeting"),
    ("help", "greeting"),
]