import sqlite3

def create_database():
    conn = sqlite3.connect('university.db')
    cursor = conn.cursor()

    # 1. Create Tables
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS courses (
        code TEXT PRIMARY KEY,
        name TEXT,
        credits INTEGER,
        dept TEXT
    )
    ''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS professors (
        name TEXT,
        email TEXT,
        dept TEXT,
        office TEXT,
        specialization TEXT
    )
    ''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS syllabus (
        course_code TEXT,
        content TEXT,
        topics TEXT,
        pdf_url TEXT,
        FOREIGN KEY(course_code) REFERENCES courses(code)
    )
    ''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS pyqs (
        course_code TEXT,
        year TEXT,
        semester TEXT,
        pdf_url TEXT,
        FOREIGN KEY(course_code) REFERENCES courses(code)
    )
    ''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS locations (
        name TEXT,
        building TEXT,
        floor INTEGER,
        hours TEXT
    )
    ''')

    # New Table for Dynamic Deadlines
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS deadlines (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT,
        description TEXT,
        due_date DATE,
        status TEXT DEFAULT 'pending'
    )
    ''')

    # 2. Populate Data (Polished Spelling)
    
    # Courses
    courses = [
        ('EMAT101L', 'Engineering Calculus', 4, 'Mathematics'),
        ('CSET301', 'Artificial Intelligence and Machine Learning', 4, 'Computer Science'),
        ('CSET243', 'Data Structures using C++', 4, 'Computer Science'),
        ('CSET201', 'Information Management Systems', 4, 'Computer Science'),
        ('CSET365', 'Web Security', 3, 'Computer Science'),
    ]
    cursor.executemany('INSERT OR IGNORE INTO courses VALUES (?,?,?,?)', courses)

    # Professors
    profs = [
        ('Dr. Vikram Singh', 'vikram.singh@university.edu', 'Computer Science', 'D-401', 'Artificial Intelligence'),
        ('Prof. Priya Sharma', 'priya.sharma@university.edu', 'Computer Science', 'A-101', 'Data Structures'),
        ('Dr. Rajesh Kumar', 'rajesh.kumar@university.edu', 'Computer Science', 'A-102', 'Automata Theory'),
    ]
    cursor.executemany('INSERT OR IGNORE INTO professors VALUES (?,?,?,?,?)', profs)

    # Syllabus
    syllabi = [
        ('CSET301', 'AI & ML', 'Search, Learning, Logic', 'pdfs/CSET301_Syllabus.pdf'),
        ('CSET243', 'Data Structures', 'Stacks, Queues, Trees', 'pdfs/CSET243_Syllabus.pdf'),
    ]
    cursor.executemany('INSERT OR IGNORE INTO syllabus VALUES (?,?,?,?)', syllabi)

    # PYQs
    pyqs = [
        ('CSET301', '2024', 'Fall', 'pdfs/CSET301_PYQ.pdf'),
        ('CSET243', '2024', 'Fall', 'pdfs/CSET243_PYQ.pdf'),
    ]
    cursor.executemany('INSERT OR IGNORE INTO pyqs VALUES (?,?,?,?)', pyqs)

    # Initial Deadlines
    deadlines = [
        ('AI Assignment 1', 'Neural Networks Basics', '2025-11-15', 'pending'),
        ('Web Sec Project', 'Vulnerability Scan', '2025-10-20', 'completed'), 
    ]
    cursor.executemany('INSERT OR IGNORE INTO deadlines (title, description, due_date, status) VALUES (?,?,?,?)', deadlines)

    # Locations
    locs = [
        ('Library', 'Main Campus Building', 2, '9 AM - 10 PM'),
        ('Cafeteria', 'Student Hub', 1, '7 AM - 9 PM'),
    ]
    cursor.executemany('INSERT OR IGNORE INTO locations VALUES (?,?,?,?)', locs)

    conn.commit()
    conn.close()
    print("Database 'university.db' created and populated successfully!")

if __name__ == '__main__':
    create_database()