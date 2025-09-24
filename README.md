# CSE440: Natural Language Processing II
## Summer 2025 - BRAC University

### 👨‍🎓 Student Information
- **Name:** Fahad Nadim Ziad
- **Student ID:** 24341216
- **Course Code:** CSE440
- **Course Title:** Natural Language Processing II
- **Section:** 02
- **Semester:** Summer 2025
- **University:** BRAC University

### 👨‍🏫 Instructor Information
- **Professor:** Dr. Farig Sadeque
- **Title:** Associate Professor
- **Department:** Computer Science and Engineering
- **University:** BRAC University
- **Google Classroom:** https://classroom.google.com/c/Nzg2MzYyNzQyNDg0?cjc=ailtmmov
- **Consultation:** https://calendar.app.google/LXSrnyDF9a7B9sXq6

---

## 📚 Course Overview

This repository contains all materials, assignments, lab work, and projects for CSE440 - Natural Language Processing II course offered during Summer 2025 at BRAC University. This is primarily an **algorithms course** with extensive programming components, focusing on practical NLP implementations rather than pure linguistics or machine learning theory.

### 🎯 Course Philosophy
**What this course IS:**
- An algorithms-focused NLP course
- Extensive programming with Python
- Practical implementation of NLP techniques
- Application of linguistics knowledge as needed

**What this course is NOT:**
- A pure linguistics course (we learn linguistics as required)
- A machine learning/neural networks course (basic refresher provided)

### 📋 Course Structure & Assessment
- **Attendance:** 0%
- **Lab Assignments:** 25%
- **Quizzes:** 15% (best 3 out of 4)
- **Midterm Exam:** 30%
- **Final Exam:** 30%

### 🔍 Why NLP is Challenging

**Ambiguity Issues:**
- **Phonetics:** "I scream" vs "Ice cream"
- **Morphology:** "Union-ized" vs "Un-ionized"
- **Syntax:** "Squad helps dog bite victim" (multiple interpretations)
- **Semantics:** "Ball" (orb vs dance)
- **Discourse:** Context-dependent meanings

**Variability:** Multiple ways to express the same concept
- "He bought it" = "He purchased it" = "He acquired it" = "It was sold to him"

**Language Evolution:** Continuous change and borrowing from other languages

### 📖 Comprehensive Course Topics

#### **Linguistics Essentials**
- Sentence Segmentation
- Tokenization
- Lemmatization/Stemming
- Parts-of-Speech (POS) Tagging
- Named Entity Recognition (NER)
- Parsing
- Coreference Resolution

#### **Machine Learning Essentials Review**
- Probability Theory Review
- Naive Bayes & Logistic Regression
- Dataset Splits, Evaluation Metrics
- Statistical Significance
- Essential ML Mathematics Refresher

#### **Text Representation**
- Representation Fundamentals
- Word Embeddings
- Contextual Embeddings

#### **Sequence Learning & Tagging**
- Sequence Tagging Basics
- Markov Models
- Deep Learning Architectures
- Recurrent Neural Networks (RNNs)

#### **Machine Translation**
- Probabilistic Translation Models
- Sequence-to-Sequence (Seq2seq) Models
- Attention Mechanisms
- Translation Challenges

#### **Advanced Topics** (Time Permitting)
- **Parsing:** Constituency & Dependency Parsing
- **Coreference Resolution**
- **Text Generation:** Encoder-Decoder Algorithms
- **Question Answering Systems**

---

## 📁 Repository Structure

```
CSE440_S25/
├── Books/                          # Reference Books and Materials
│   ├── 100 essentials NLP.pdf
│   ├── Jacob Eisenstein - Introduction to Natural Language Processing (NLP)-MIT Press.pdf
│   ├── jurafsky book.pdf
│   ├── Manning C.D., Schütze H. - Foundations of Statistical Natural Language Processing (1999).pdf
│   └── NLTK.pdf
├── LAB/                           # Laboratory Assignments and Work
│   ├── CSE440 Lab Outline _ Sum'25.pdf
│   ├── LAB1(NLTK_EDA)/           # Lab 1: NLTK Exploratory Data Analysis
│   │   ├── 24341216_FahadNadimZiad_CSE440_02_Lab1_Summer2025.py
│   │   ├── assignment.ipynb
│   │   ├── CSE440_Sec2_Lab_Assignment1_Summer2025 _ Student Version.pdf
│   │   └── nltk.ipynb
│   ├── LAB2/                     # Lab 2: Text Processing and Analysis
│   │   ├── 24341216_Fahad Nadim Ziad_CSE440_02_Lab_Assignment2_Summer2025.ipynb
│   │   ├── CSE440_Sec2_Lab_Assignment2_Summer2025 _ Student Version - Google Docs.pdf
│   │   ├── glove.6B.100d.txt
│   │   ├── IMDB Dataset.csv
│   │   ├── task1.ipynb
│   │   ├── task2.ipynb
│   │   └── task3.ipynb
│   ├── LAB3/                     # Lab 3: Advanced NLP Techniques
│   │   ├── 24341216_Fahad Nadim Ziad_CSE440_02_Lab_Assignment3_Summer2025.ipynb
│   │   └── CSE440_Sec2_Lab_Assignment3_Summer2025 _ Student Version.docx - Google Docs.pdf
│   └── LAB4/                     # Lab 4: Current Assignment
│       └── CSE440_Sec2_Lab_Assignment4_Summer2025 _ Student Version.docx.pdf
├── Lecture/                       # Lecture Materials
│   ├── pdfs/                     # PDF versions of slides
│   │   ├── 1_Introduction.pptx.pdf
│   │   ├── 2_linguistics_essentials.pptx.pdf
│   │   ├── 3_ML_essentials.pptx.pdf
│   │   ├── 4_word_representation.pptx.pdf
│   │   ├── 5_sequence_learning.pptx.pdf
│   │   ├── 6_nn_rnn.pptx.pdf
│   │   └── 7_translation.pptx.pdf
│   └── pptx/                     # PowerPoint presentations
│       ├── 1_Introduction.pptx
│       ├── 2_linguistics_essentials.pptx
│       ├── 3_ML_essentials.pptx
│       ├── 4_word_representation.pptx
│       ├── 5_sequence_learning.pptx
│       ├── 6_nn_rnn.pptx
│       └── 7_translation.pptx
├── Outline/                       # Course Outline and Syllabus
├── project/                       # Course Project
│   ├── [Updated] Question Answer Classification Dataset[Test].csv
│   ├── Question Answer Classification Dataset 1[Training].csv
│   ├── glove.6B.100d.txt
│   └── Summer 2025 - CSE440 Lab Project - Google Docs.pdf
└── Questions/                     # Exam Questions and Materials
    ├── finals/
    │   ├── q1.pdf
    │   └── q2.pdf
    └── mid/
        ├── makeupmid_summer23.pdf
        ├── mid_fall24.pdf
        ├── mid_spring25.pdf
        └── mid_summer23.pdf
```

---

## 🔬 Laboratory Work

### Lab 1: NLTK Exploratory Data Analysis
- **Focus:** Introduction to NLTK library and basic text processing
- **Technologies:** Python, NLTK, Jupyter Notebook
- **Key Concepts:** Tokenization, POS tagging, Named Entity Recognition

### Lab 2: Text Processing and Analysis
- **Focus:** Advanced text processing and word embeddings
- **Technologies:** Python, GloVe embeddings, IMDB dataset
- **Key Concepts:** Word vectors, sentiment analysis, text classification

### Lab 3: Advanced NLP Techniques
- **Focus:** Deep learning approaches to NLP
- **Technologies:** Neural networks, advanced preprocessing
- **Key Concepts:** Sequence modeling, feature engineering

### Lab 4: [Current Assignment]
- **Status:** In Progress
- **Focus:** [To be updated based on assignment requirements]

---

## 🎯 Course Project

**Project Title:** Question Answer Classification

**Dataset:**
- Training Dataset: Question Answer Classification Dataset 1[Training].csv
- Test Dataset: [Updated] Question Answer Classification Dataset[Test].csv
- Word Embeddings: GloVe 6B.100d embeddings

**Objective:** Develop a machine learning model to classify question-answer pairs and improve understanding of automated question answering systems.

---

## 📖 Reference Materials

The course utilizes several authoritative texts in Natural Language Processing:

1. **Jacob Eisenstein** - Introduction to Natural Language Processing (MIT Press)
2. **Jurafsky & Martin** - Speech and Language Processing (Jurafsky book)
3. **Manning & Schütze** - Foundations of Statistical Natural Language Processing (1999)
4. **100 Essentials NLP** - Comprehensive NLP guide
5. **NLTK Documentation** - Natural Language Toolkit reference

---

## 💻 Technologies & Tools Used

- **Programming Language:** Python
- **Libraries:** NLTK, NumPy, Pandas, Scikit-learn, Matplotlib, Seaborn
- **Development Environment:** Jupyter Notebook, Python IDEs
- **Word Embeddings:** GloVe (Global Vectors for Word Representation)
- **Datasets:** IMDB Movie Reviews, Custom QA Classification datasets
- **Version Control:** Git & GitHub

---

## 🏆 Learning Outcomes & Prerequisites

### Prerequisites
- **Programming Experience:** Several semesters of programming experience
- **Primary Language:** Python proficiency required
- **Mathematics:** Ability to understand mathematical notation (helpful but not required)
- **Linguistics:** Helpful but not required (learned as needed)
- **Machine Learning:** Basic understanding helpful (refresher provided)

By the end of this course, students will be able to:

1. **Implement core NLP algorithms** for text processing and analysis
2. **Apply linguistic processing techniques** including tokenization, POS tagging, and NER
3. **Develop machine learning models** for various NLP tasks
4. **Work with word embeddings** and contextual representations
5. **Build sequence tagging systems** using Markov models and RNNs
6. **Create translation systems** using seq2seq models and attention mechanisms
7. **Handle NLP challenges** including ambiguity, variability, and language change
8. **Evaluate NLP systems** using appropriate metrics and statistical significance testing
9. **Understand parsing techniques** for syntactic analysis
10. **Develop end-to-end NLP applications** for real-world problems

---

## 📊 Assessment Structure

| Component | Weight | Details |
|-----------|--------|----------|
| **Laboratory Assignments** | 25% | Hands-on programming and NLP algorithm implementation |
| **Quizzes** | 15% | Best 3 out of 4 quizzes (regular assessment of concepts) |
| **Midterm Examination** | 30% | Theoretical understanding and practical application |
| **Final Examination** | 30% | Comprehensive assessment of all course material |
| **Attendance** | 0% | Not graded but highly recommended for success |

### Key Projects & Assignments
- **Course Project:** Question Answer Classification system development
- **Lab Work:** Progressive implementation of NLP algorithms and techniques
- **Regular Quizzes:** Continuous assessment of theoretical and practical knowledge

---

## 🎓 Class Resources & Support

### 📚 Google Classroom
- **Primary Hub:** All notifications, course content, books, lectures, and recordings
- **Classroom Link:** https://classroom.google.com/c/Nzg2MzYyNzQyNDg0?cjc=ailtmmov
- **Access:** Use the QR code provided in class or join via the link above

### 🕐 Professor Consultation Hours
- **Instructor:** Dr. Farig Sadeque
- **Booking Required:** Preferred appointment-based consultation
- **Schedule Appointment:** https://calendar.app.google/LXSrnyDF9a7B9sXq6
- **Availability:** Regular consultation hours (booking recommended)

### 💡 Academic Support
- **Office Hours:** Available through appointment booking system
- **Course Questions:** Use Google Classroom for course-related queries
- **Technical Support:** Lab instructors available during lab sessions

---

## 🤝 Contributing

This repository is for academic purposes. If you're a fellow student or researcher interested in NLP, feel free to explore the code and methodologies used.

---

## 📧 Contact Information

### Student Contact
**Fahad Nadim Ziad**
- Student ID: 24341216
- University: BRAC University
- Course: CSE440 - Natural Language Processing II
- Semester: Summer 2025

### Course Instructor
**Dr. Farig Sadeque**
- Associate Professor, Computer Science and Engineering
- BRAC University
- Consultation: https://calendar.app.google/LXSrnyDF9a7B9sXq6
- Google Classroom: https://classroom.google.com/c/Nzg2MzYyNzQyNDg0?cjc=ailtmmov

---

## 📄 License

This repository is created for educational purposes as part of coursework at BRAC University. All materials are used in accordance with academic guidelines and fair use policies.

---

**Note:** This repository represents the complete journey through CSE440 - Natural Language Processing II course, documenting learning progress, practical implementations, and theoretical understanding gained throughout the semester.