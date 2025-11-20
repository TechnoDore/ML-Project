# dataset.py
# Contains training sentences for the ML model.

TRAINING_QUERIES = [
    # Syllabus
    ("syllabus for Engineering Calculus", "syllabus"), 
    ("what is covered in CSET243", "syllabus"),
    ("course content for Information Management Systems", "syllabus"), 
    ("topics in High Performance Computing", "syllabus"),
    ("what do I need to study for CSET201", "syllabus"),
    ("what should I prepare for CSET301", "syllabus"),
    ("what to read for Exam", "syllabus"),

    # Professor
    ("who teaches Data Structures", "professor"), 
    ("contact details for the UI/UX professor", "professor"), 
    ("who is the instructor for Automata Theory", "professor"), 
    ("where does Dr. Vikram sit", "professor"), 
    ("where is the office of Prof Sharma", "professor"), 
    ("cabin location of the HOD", "professor"), 
    ("professor's room number", "professor"),
    ("where can I find the teacher", "professor"),

    # PYQ
    ("past year questions for Probability and Statistics", "pyq"), 
    ("previous exam papers for CSET201", "pyq"),
    ("old questions for data structures", "pyq"), 
    ("old papers for maths", "pyq"),
    ("question paper for physics", "pyq"),
    
    # Deadline (READING)
    ("when is the assignment due", "deadline"), 
    ("upcoming deadlines", "deadline"), 
    ("exam dates", "deadline"), 
    ("is there any work pending", "deadline"),
    ("what assignments are left", "deadline"),
    ("show passed deadlines", "deadline_history"),
    ("what deadlines are completed", "deadline_history"),
    ("previous deadlines", "deadline_history"),

    # Deadline (SETTING & MARKING)
    ("add deadline", "add_deadline_intent"),
    ("set a new deadline", "add_deadline_intent"),
    ("I want to add a task", "add_deadline_intent"),
    ("mark task as done", "mark_deadline_intent"),
    ("mark assignment as completed", "mark_deadline_intent"),

    # Location
    ("where is the library", "location"), 
    ("how to get to the computer lab", "location"),
    ("location of cafeteria", "location"),

    # Greeting
    ("hi", "greeting"), 
    ("hello", "greeting"),
    ("help", "greeting"),
]