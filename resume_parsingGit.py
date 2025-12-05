import os
import re
import tempfile
import traceback
from pathlib import Path
from typing import List, Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import json
import re
import ast
from typing import List
import threading
from TTS.api import TTS
import sounddevice as sd
from docxtpl import DocxTemplate
from docx2pdf import convert
import pythoncom
import simpleaudio as sa
from flask import Flask, request, jsonify, render_template
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import desc, func

# PDF/DOCX parsing
import pdfplumber
import docx
from flask_sock import Sock




os.environ["GEMINI_API_KEY"] = "Your gemini api key"
# Attempt LangChain / LangGraph imports (optional)
USE_LANGCHAIN = False
USE_LANGGRAPH = False
try:
    from langchain import PromptTemplate as LC_PromptTemplate  # just to detect presence
    from langchain.chains import LLMChain
    from langchain.llms.base import LLM
    USE_LANGCHAIN = True
except Exception:
    LC_PromptTemplate = None

try:
    import langgraph as lg
    USE_LANGGRAPH = True
except Exception:
    lg = None

# Embeddings (optional)
USE_EMBEDDINGS = True
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
EMBEDDER = None
cosine_similarity = None
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    EMBEDDER = SentenceTransformer(EMBEDDING_MODEL)
except Exception:
    EMBEDDER = None
    USE_EMBEDDINGS = False

# Gemini (Google Generative AI) detection
MODEL = None
try:
    import google.generativeai as genai
    GEMINI_KEY = os.getenv("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY") or None
    if GEMINI_KEY:
        genai.configure(api_key=GEMINI_KEY)
        GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
        try:
            MODEL = genai.GenerativeModel(GEMINI_MODEL)
        except Exception:
            MODEL = None
except Exception:
    MODEL = None

# Flask + Postgres config
app = Flask(__name__)
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:yogesh@localhost:5433/postgres")
app.config["SQLALCHEMY_DATABASE_URI"] = DATABASE_URL
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)



# ---------------- Models ----------------
class Candidate(db.Model):
    __tablename__ = "candidates"
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(400))
    email = db.Column(db.String(300), index=True)
    phone = db.Column(db.String(100))
    skills = db.Column(db.Text)  # comma separated
    qualifications = db.Column(db.Text)
    achievements = db.Column(db.Text)
    work_experience_years = db.Column(db.Float)
    summary_text = db.Column(db.Text)
    embedding = db.Column(db.Text)  # store repr list
    score = db.Column(db.Float)
    resume_filename = db.Column(db.String(500))
    parsed_at = db.Column(db.DateTime, server_default=func.now())
    status = db.Column(db.String(50), default="processed")
    job_role=db.Column(db.Text)
    company_name=db.Column(db.Text)
    salary=db.Column(db.Float)

class MatchRecord(db.Model):
    __tablename__ = "matchrecords"
    id = db.Column(db.Integer, primary_key=True)
    candidate_id = db.Column(db.Integer, db.ForeignKey("candidates.id"))
    skill_matches = db.Column(db.Text)
    score = db.Column(db.Float)
    created_at = db.Column(db.DateTime, server_default=func.now())

class EmailLog(db.Model):
    __tablename__ = "emaillogs"
    id = db.Column(db.Integer, primary_key=True)
    candidate_id = db.Column(db.Integer, db.ForeignKey("candidates.id"), nullable=True)
    to_email = db.Column(db.String(400))
    subject = db.Column(db.String(500))
    body = db.Column(db.Text)
    sent_ok = db.Column(db.Boolean, default=False)
    attempt_at = db.Column(db.DateTime, server_default=func.now())
    error = db.Column(db.Text)

class InterviewSession(db.Model):
    __tablename__ = "interviews"
    id = db.Column(db.Integer, primary_key=True)
    candidate_id = db.Column(db.Integer, db.ForeignKey("candidates.id"))
    started_at = db.Column(db.DateTime, server_default=func.now())
    ended_at = db.Column(db.DateTime)
    transcript = db.Column(db.Text)  # full Q&A
    score = db.Column(db.Float)      # AI-calculated

class Offeraccepted(db.Model):
    __tablename__ = "offeraccepted"
    id=db.Column(db.Integer, primary_key=True)
    candidate_id=db.Column(db.Integer, db.ForeignKey("candidates.id"))
    job_role=db.Column(db.Text)
    job_type=db.Column(db.Text)
    candidate_name= db.Column(db.Text)


class ConversationHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    candidate_id = db.Column(db.String(20))
    role = db.Column(db.String(10))  # 'user' / 'assistant'
    content = db.Column(db.Text)

class DocumentTemplate(db.Model):
    __tablename__ = "document_templates"
    id = db.Column(db.Integer, primary_key=True)
    template_name = db.Column(db.String(200))
    company_name = db.Column(db.String(200))
    file_path = db.Column(db.String(300))
    uploaded_at = db.Column(db.DateTime, server_default=func.now())


class InternshipProcess(db.Model):
    __tablename__ = "internship_process"
    id = db.Column(db.Integer, primary_key=True)
    candidate_id = db.Column(db.Integer, db.ForeignKey("candidates.id"))
    internship_end = db.Column(db.DateTime)
    review_status = db.Column(db.String(50), default="pending")  # pending / approved / rejected
    approved_doc = db.Column(db.String(200), nullable=True)
    last_updated = db.Column(db.DateTime, server_default=func.now(), onupdate=func.now())

class InternshipProcess1(db.Model):
    __tablename__="internship_process1"
    id = db.Column(db.Integer, primary_key=True)
    candidate_id = db.Column(db.Integer, nullable=False)
    review_status = db.Column(db.String(50), default="pending")
    # maybe your columns are named differently
    internship_start = db.Column(db.Date)
    internship_end= db.Column(db.Date)


with app.app_context():
    db.create_all()

# ---------------- Utilities ----------------
EMAIL_RE = re.compile(r"[\w\.-]+@[\w\.-]+\.\w+")
PHONE_RE = re.compile(r'(\+?\d{1,3}[\s\-.]?)?(\(?\d{2,5}\)?[\s\-.]?)?\d{5,10}')

def extract_text_from_pdf(path: str) -> str:
    parts = []
    try:
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                try:
                    t = page.extract_text(x_tolerance=2, y_tolerance=2) or ""
                    if not t.strip():
                        chars = page.chars
                        chars = sorted(chars, key=lambda c: (c["top"], c["x0"]))
                        lines = {}
                        for c in chars:
                            top = round(c["top"])
                            lines.setdefault(top, [])
                            lines[top].append(c["text"])
                        t = "\n".join("".join(lines[top]) for top in sorted(lines))
                    parts.append(t)
                except Exception:
                    parts.append("")
    except Exception:
        app.logger.exception("PDF parse failed for %s", path)
    return "\n".join(parts)

def extract_text_from_docx(path: str) -> str:
    try:
        d = docx.Document(path)
        return "\n".join([p.text for p in d.paragraphs])
    except Exception:
        app.logger.exception("DOCX parse failed for %s", path)
        return ""

# Prompts (LangChain prompt compatibility)
try:
    PromptTemplate = LC_PromptTemplate if LC_PromptTemplate else None
except Exception:
    PromptTemplate = None

EXTRACTION_PROMPT_TEXT = """
    You are a precise extractor. From the resume text below produce a single JSON object with these keys:
    provide the output in between double dollar like $$...$$:
    $$
    {{
    "Name": string or null,
    "Email": string or null,
    "Phone": string or null,
    "Skills": [ "skill1", "skill2", ... ],
    "Qualifications": [ "qual1", "qual2", ... ],
    "TotalExperienceYears": number or null,
    "WorkExperience": string or null,
    "Achievements": [ "achievement1", ... ],
    "Summary": string or null,
    "JobRole": string,
    "CompanyName": string,
    "Salary":integer
    }}
    $$
    
    Guidelines:
    - Return ONLY valid JSON (no surrounding text). Use plain strings and numbers.
    - "Skills" must include all skills explicitly mentioned in the resume (do not invent).
    - If a field is not present in the resume, use null (or empty list for arrays).
    - Keep values short and factual.

    Job Description: {job_description}

    Resume:
    {resume_text}
"""

SCORE_PROMPT_TEXT = """
Return a single number 0-100 estimating candidate fit.

Candidate skills: {skills}
Candidate experience (years): {experience}
Job description: {job_description}

Return only the numeric score (e.g. 78)
"""

def call_gemini_text(prompt_text: str, max_output_tokens: int = 512) -> str:
    if MODEL is None:
        print("[Debug] gemini model is none")
        app.logger.info("Gemini MODEL not configured — skipping LLM calls.")
        return ""
    print("\n[DEBUG] Sending prompt to Gemini...")
    print(prompt_text[:500], "...\n")
    try:
        resp = MODEL.generate_content(
            prompt_text,
            generation_config={"max_output_tokens": max_output_tokens}
        )
        print("[Debug] Raw gemini response:", resp)
        if hasattr(resp, "text") and resp.text:
            print("[Debug] gemini response Text:", resp.text)
            return resp.text.strip()
        if hasattr(resp, "candidates"):
            texts = []
            for c in resp.candidates:
                if hasattr(c, "content"):
                    if isinstance(c.content, str):
                        texts.append(c.content)
                    elif hasattr(c.content, "text"):
                        texts.append(c.content.text)
            return "\n".join(texts).strip()
        return str(resp)
    except Exception as e:
        print("[DEBUG] ❌ Gemini API FAILED")
        print("[DEBUG] Error:", e)
        app.logger.exception("Gemini call failed: %s", str(e))
        return ""

def gemini_extract(resume_text: str, job_description: str ="") -> dict:
    print("=======Gemini Extract Start=========")
    prompt_text = EXTRACTION_PROMPT_TEXT.format(resume_text=resume_text, job_description=job_description)
    out = call_gemini_text(prompt_text, max_output_tokens=3000)
    print("watch this :",out)
    match = re.search(r'\$\$(.*?)\$\$', out, re.DOTALL) 
    if match: 
        print("***match found**", out) 
        out = match.group(1)
    match = re.search(r'json\s*(.*)', out, re.DOTALL) 
    if match: 
        out = match.group(1) # Remove any trailing triple backticks if they exist 
    out = re.sub(r'```', '', out).strip()
    parsed = json.loads(out)
    # parsed = parse_extraction_output(out)
    return parsed

def gemini_score(skills: str, experience: str, jobdesc: str) -> float:
    prompt_text = SCORE_PROMPT_TEXT.format(skills=skills, experience=experience or "0", job_description=jobdesc or "")
    out = call_gemini_text(prompt_text, max_output_tokens=50)
    if not out:
        return 0.0
    m = re.search(r"(\d+(\.\d+)?)", out)
    if m:
        v = float(m.group(1))
        return max(0.0, min(100.0, v))
    return 0.0


def semantic_similarity_local(vec1, vec2):
    try:
        return float((cosine_similarity([vec1], [vec2])[0][0] + 1) / 2)
    except Exception:
        return 0.0

def compute_score(candidate_skills: List[str], cand_exp: float, job_required: List[str], job_desc: str) -> float:
    req = set([s.lower().strip() for s in job_required if s.strip()])
    cand = set([s.lower().strip() for s in candidate_skills if s.strip()])
    skill_match = 0.0
    if req:
        skill_match = len(req & cand) / len(req)
    sem_sim = 0.0
    if USE_EMBEDDINGS and EMBEDDER and job_desc:
        try:
            job_emb = EMBEDDER.encode(job_desc[:3000])
            cand_emb = EMBEDDER.encode(" ".join(candidate_skills)[:3000])
            sem_sim = semantic_similarity_local(job_emb.tolist(), cand_emb.tolist())
        except Exception:
            sem_sim = 0.0
    cand_exp = float(cand_exp)
    exp_score = min(cand_exp / max(1.0, 1.0), 1.0) if cand_exp else 0.0
    final = 0.5 * skill_match + 0.3 * sem_sim + 0.2 * exp_score
    return float(final * 100)

# Email send + log with retry
import smtplib
from email.message import EmailMessage
SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 587
SMTP_USER = "nsyogeshraj@gmail.com"
SMTP_PASS = "Your google app keyword"
FROM_EMAIL = SMTP_USER

def generate_email_llm(candidate_name: str, company_name: str, job_role: str, candidate_id: int,token=None,session_id=None) -> tuple[str, str]:
    # Build real-time interview URL dynamically based on current host
    HOST = "localhost:8000"
    scheduling_link = f"http://{HOST}/"

    prompt = f"""
    You are an HR assistant. 
    Write a professional and friendly email subject and body inviting the candidate {candidate_name} 
    for an interview in the company {company_name} for the position of {job_role}. 
    Include the scheduling link: {scheduling_link} Your session ID: {session_id}
    Your token: {token}.
    Return only JSON with "subject" and "body".
    """
    response = call_gemini_text(prompt, max_output_tokens=256)

    import json
    try:
        out = response
        match = re.search(r'\$\$(.*?)\$\$', out, re.DOTALL)
        if match:
            out = match.group(1)
        match = re.search(r'json\s*(.*)', out, re.DOTALL)
        if match:
            out = match.group(1)
        out = re.sub(r'```', '', out).strip()
        parsed = json.loads(out)

        subject = parsed.get("subject", f"Interview Invitation — {job_role}")
        body = parsed.get("body", f"Hi {candidate_name}, please schedule your interview here: {scheduling_link}")
    except Exception:
        subject = f"Interview Invitation — {job_role}"
        body = f"Hi {candidate_name}, please schedule your interview here: {scheduling_link}"

    return subject, body


tts= TTS("tts_models/en/ljspeech/tacotron2-DDC")
def speak(text:str):
    audio=tts.tts(text)
    sd.play(audio, samplerate=22050)
    sd.wait()


import speech_recognition as sr

recognizer = sr.Recognizer()

def listen_until_silence(timeout=3, phrase_time_limit=12):
    """
    Real-time STT that listens continuously until the user becomes silent
    for 'timeout' seconds.
    """

    with sr.Microphone() as source:
        print("\n🎤 Listening... (stop speaking to finish)\n")

        recognizer.adjust_for_ambient_noise(source, duration=0.5)

        audio_data = recognizer.listen(
            source,
            timeout=timeout,            # stop if user is silent
            phrase_time_limit=phrase_time_limit  # max length of one phrase
        )

        try:
            text = recognizer.recognize_google(audio_data)
            print("📝 Recognized:", text)
            return text

        except sr.WaitTimeoutError:
            print("⌛ No speech detected (timeout).")
            return ""

        except sr.UnknownValueError:
            print("❓ Could not understand audio.")
            return ""

        except Exception as e:
            print("❌ STT Error:", e)
            return ""

def generate_personalized_questions(resume_text: str, job_desc: str, num_questions: int = 5) -> List[str]:
    """
    Generate a list of interview questions for a candidate based on their resume and job description.
    Uses Gemini LLM and speaks each question using TTS.
    """
    prompt = f"""
    You are an HR interviewer. Generate {num_questions} concise interview questions for a candidate.
    Use the candidate's resume and the job description below.
    Resume: {resume_text[:2000]}
    Job Description: {job_desc[:2000]}
    Return in list format like:
    return the output in between double dollar like $$...$$:
    $$
    ['question_1',..., 'question_n']
    $$
    """
    response = call_gemini_text(prompt, max_output_tokens=2000)

    try:
        match = re.search(r'\$\$(.*?)\$\$', response, re.DOTALL)
        out = match.group(1) if match else response
        questions = ast.literal_eval(out)
        return questions

    except:
        # Fallback simple questions
        fallback = [
            f"Tell me about your experience with {skill}"
            for skill in re.findall(r'\b\w+\b', job_desc)[:num_questions]
        ]
        return fallback


def score_interview(transcript: List[str], resume_text: str, job_desc: str) -> float:
    """
    Score candidate's interview based on transcript vs JD and resume.
    Returns 0-100 score.
    """
    # answers_text = " ".join(transcript)
    prompt = f"""
        Evaluate the candidate's answers below and provide a score 0-100 for suitability for the role.
        Resume: {resume_text[:5000]}
        Job Description: {job_desc}
        Candidate Question and Answers: {transcript}
        Return only a numeric score.
    """
    response = call_gemini_text(prompt, max_output_tokens=50)
    m = re.search(r"(\d+(\.\d+)?)", response)
    if m:
        return float(m.group(1))
    return 0.0

# ---------------- WebSocket Interview ----------------



def send_invite_email_and_log(
    candidate_id: Optional[int],
    to_email: str,
    candidate_name: str,
    job_role: str,
    company_name:str,
    scheduling_link: str = "#",
    max_retries: int = 2,
    token: Optional[str] = None,
    session_id: Optional[int] = None,
):
    """
    Sends an LLM-generated interview invite email and logs the attempt in the DB
    """
    subject, body = generate_email_llm(candidate_name, job_role,company_name, scheduling_link)
    if scheduling_link:
        body += f"\n\nSchedule Link: {scheduling_link}"

# ✔ Session ID separate
    if session_id:
        body += f"\nSession ID: {session_id}"
    # Log initial attempt
    log = EmailLog(
        candidate_id=candidate_id,
        to_email=to_email,
        subject=subject,
        body=body,
        sent_ok=False,
        error=None,
        attempt_at=datetime.utcnow()
    )
    db.session.add(log)
    db.session.commit()

    ok = False
    err = None

    # Send email via SMTP
    for attempt in range(max_retries + 1):
        try:
            if not SMTP_HOST or not SMTP_USER or not SMTP_PASS:
                raise RuntimeError("SMTP credentials not configured in environment")
            
            msg = EmailMessage()
            msg["Subject"] = subject
            msg["From"] = FROM_EMAIL
            msg["To"] = to_email
            msg.set_content(body)

            with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=30) as server:
                server.starttls()
                server.login(SMTP_USER, SMTP_PASS)
                server.send_message(msg)

            ok = True
            err = None
            break

        except Exception as e:
            err = str(e)

    # Update log after attempt
    log.sent_ok = ok
    log.error = err
    log.attempt_at = datetime.utcnow()
    db.session.add(log)
    db.session.commit()

    return ok, err

# ---------------- API / Processing ----------------

def send_offer_letter_and_log(
    candidate_id: int,
    to_email: str,
    candidate_name: str,
    job_role: str,
    offer_file_path: str,
    company_name: str,
    max_retries: int = 2
   
):
    """
    Sends an offer letter email with attachment (PDF/DOCX)
    and logs it in EmailLog table.
    """

    # Build subject & body (can replace with LLM if needed)
    subject = f"Congratulations {candidate_name}! Your Offer Letter for {job_role}"
    body = (
    f"Dear {candidate_name},\n\n"
    f"We are pleased to inform you that, based on your strong performance in the interview "
    f"process, you have been selected for the position of {job_role} with {company_name}.\n\n"
    f"Please find your official offer letter attached to this email. "
    f"Kindly review the document carefully and confirm your acceptance by replying to this email "
    f"or signing and returning the attached offer letter within the specified timeline.\n\n"
    f"We are confident that your skills and experience will greatly contribute to our organization’s success, "
    f"and we look forward to welcoming you onboard.\n\n"
    f"In case of any queries or further clarification, please feel free to contact the HR department.\n\n"
    f"Regards,\n"
    f"Human Resources Department\n"
    f"{company_name}"
)


    # Log first attempt
    log = EmailLog(
        candidate_id=candidate_id,
        to_email=to_email,
        subject=subject,
        body=body,
        sent_ok=False,
        error=None,
        attempt_at=datetime.utcnow()
    )
    db.session.add(log)
    db.session.commit()

    ok = False
    err = None

    for attempt in range(max_retries + 1):
        try:
            if not SMTP_HOST or not SMTP_USER or not SMTP_PASS:
                raise RuntimeError("SMTP credentials not configured in environment")

            msg = EmailMessage()
            msg["Subject"] = subject
            msg["From"] = FROM_EMAIL
            msg["To"] = to_email
            msg.set_content(body)

            # Attach the offer letter
            with open(offer_file_path, "rb") as f:
                file_data = f.read()
                filename = os.path.basename(offer_file_path)

                # Detect file type automatically
                if filename.endswith(".pdf"):
                    msg.add_attachment(file_data, maintype="application", subtype="pdf", filename=filename)
                elif filename.endswith(".docx"):
                    msg.add_attachment(file_data, maintype="application",
                                       subtype="vnd.openxmlformats-officedocument.wordprocessingml.document",
                                       filename=filename)
                else:
                    # fallback – treat as octet-stream
                    msg.add_attachment(file_data, maintype="application",
                                       subtype="octet-stream", filename=filename)

            # SMTP Send
            with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=30) as server:
                server.starttls()
                server.login(SMTP_USER, SMTP_PASS)
                server.send_message(msg)

            ok = True
            break

        except Exception as e:
            err = str(e)

    # Update email log
    log.sent_ok = ok
    log.error = err
    log.attempt_at = datetime.utcnow()
    db.session.add(log)
    db.session.commit()

    return ok, err

@app.route("/")
def ui():
    return render_template("index.html") if Path("templates/index.html").exists() else jsonify({"status":"resume-scanner","version":"upgraded"})

def process_single_resume(path: str, filename: str, job_desc: str, required_skills: List[str]):
    # returns dict result
    with app.app_context():
        try:
            text = ""
            if filename.lower().endswith(".pdf"):
                text = extract_text_from_pdf(path)
            elif filename.lower().endswith(".docx"):
                text = extract_text_from_docx(path)
            else:
                text = extract_text_from_docx(path) or extract_text_from_pdf(path)
            if not text:
                return {"error": "no_text", "filename": filename}

            parsed = gemini_extract(text, job_desc)

            skills_set = parsed.get("Skills",[])
            skills_list = sorted(list(skills_set))

            # match to job description / jobrole
            matched_skills = []
            jd_lower = job_desc.lower() if job_desc else ""
            for skill in skills_list:
                if skill.lower() in jd_lower:
                    matched_skills.append(skill)

            # experience parse
            exp_y = 0.0
            exp_y = parsed.get("TotalExperienceYears",0) 
            if not exp_y:
                exp_y = 0
            
            emb = None
            if USE_EMBEDDINGS and EMBEDDER:
                try:
                    emb_vec = EMBEDDER.encode((parsed.get("Summary") or text)[:3000])
                    emb = emb_vec.tolist()
                except Exception:
                    emb = None

            # optional gemini scoring
            gem_score = None
            if MODEL and job_desc:
                try:
                    gem_score = gemini_score(",".join(skills_list), str(exp_y), job_desc)
                except Exception:
                    gem_score = None

            # compute comp_score (jobrole or provided required_skills)
            comp_score = compute_score(skills_list, exp_y, required_skills, job_desc)

            final_score = comp_score
            if gem_score is not None:
                final_score = (comp_score + gem_score) / 2.0

            status = "processed"
            # save candidate into DB (app context ensures thread-safety)
            cand = Candidate(
                name = parsed.get("Name"),
                email = parsed.get("Email"),
                phone = parsed.get("Phone"),
                skills = ",".join(skills_list),
                qualifications = parsed.get("Qualifications"),
                achievements = parsed.get("Achievements"),
                work_experience_years = float(exp_y or 0.0),
                summary_text = (parsed.get("Summary") or text)[:4000],
                embedding = str(emb) if emb is not None else None,
                score = final_score,
                resume_filename = filename,
                status=status,
                job_role=parsed.get("JobRole"),
                company_name=parsed.get("CompanyName"),
                salary=parsed.get("Salary")


            )
            db.session.add(cand)
            db.session.commit()

            if cand.score >= 70 and cand.email:
                start_interview_resp = start_interview(cand.id)
                interview_data = start_interview_resp.get_json() if hasattr(start_interview_resp, "get_json") else start_interview_resp
                session_id = interview_data.get('session_id')
                token = interview_data.get('token')
                scheduling_link = f"http://localhost:8000/start-interview/{cand.id}?token={interview_data.get('token')}"
                send_invite_email_and_log(
                    candidate_id=cand.id,
                    to_email=cand.email,
                    candidate_name=cand.name,
                    job_role=cand.job_role,
                    company_name=cand.company_name,
                    token=interview_data.get('token'),
                    session_id=interview_data.get('session_id'),  # NEW ✔
                    scheduling_link=scheduling_link 
                )

            mr = MatchRecord(candidate_id=cand.id,  skill_matches=",".join(matched_skills), score=final_score)
            db.session.add(mr)
            db.session.commit()
            print("11111")
            return {
                "id": cand.id,
                "name": cand.name,
                "phone": cand.phone,
                "email": cand.email,
                "summary_text": cand.summary_text,
                "skills": skills_list,
                "matched_skills": matched_skills,
                "experience_years": exp_y,
                "score": round(final_score,2),
                "resume_filename": filename
            }
        except Exception as e:
            app.logger.exception("Processing failed for %s: %s", filename, str(e))
            return {"error":"exception","filename": filename, "detail": str(e)}

@app.route("/upload-resumes", methods=["POST"])
def upload_resumes():
    app.logger.info("Upload request received")
    files = request.files.getlist("resumes")
    if not files:
        return jsonify({"error":"No files (field name 'resumes')"}), 400
    if len(files) > 101:
        return jsonify({"error":"Too many files (limit 100)"}), 400

    job_desc = request.form.get("job_description","") or ""
    required_skills_input = request.form.get("required_skills","") or ""
    required_skills = [x.strip() for x in required_skills_input.split(",") if x.strip()]

    top_k = int(request.form.get("top_k",5))
    send_invites_flag = request.form.get("send_invites","false").lower() in ("true","1","yes")
    
    threshold = float(request.form.get("threshold", 70.0))
    workers = int(request.form.get("workers", min(6, max(2, (os.cpu_count() or 2)))))

    tmpdir = Path(tempfile.mkdtemp(prefix="resumes_"))
    results = []
    print("0000")

    # save files and prepare tasks
    saved = []
    for idx, f in enumerate(files, start=1):
        filename = f.filename or f"{idx}.bin"
        p = tmpdir / filename
        f.save(str(p))
        saved.append((str(p), filename))

    # concurrent processing
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(process_single_resume, path, name, job_desc, required_skills) for path, name in saved]
        for fut in as_completed(futures):
            try:
                res = fut.result()
                if res:
                    results.append(res)
            except Exception as e:
                app.logger.exception("Worker exception: %s", str(e))

    # sort & shortlist
    sorted_by = sorted([r for r in results if not r.get("error")] , key=lambda x: x.get("score",0.0), reverse=True)
    shortlisted = sorted_by[:top_k]
    shortlist_ids = {c["id"] for c in shortlisted}

    for c in sorted_by:
        # default status
        c["status"] = "shortlisted" if (c["id"] in shortlist_ids or c.get("score",0.0) >= threshold) else "not_shortlisted"
        # enforce minimum pass
        if c.get("score",0.0) < 55:
            c["status"] = "not_shortlisted"
        c["message"] = ("Selected for interview" if c["status"] == "shortlisted" else "Not selected in this screening round")

    # send invites optionally and log
    invites_sent = []
    if send_invites_flag:
        for c in sorted_by:
            if c.get("status") == "shortlisted" and c.get("email"):
                try:
                    ok, err = send_invite_email_and_log(c.get("id"), c["email"], c.get("name"), job_role="Applied Role")
                    c["invite_sent"] = ok
                    invites_sent.append({"id": c.get("id"), "email": c.get("email"), "sent": ok, "err": err})
                except Exception as e:
                    app.logger.exception("Failed sending invite: %s", str(e))

    # cleanup tmp files
    try:
        for p,_ in saved:
            try:
                Path(p).unlink()
            except Exception:
                pass
        tmpdir.rmdir()
    except Exception:
        pass

    return jsonify({
        "processed": len(results),
        "shortlisted_count": len([c for c in sorted_by if c["status"]=="shortlisted"]),
        "shortlisted": [c for c in sorted_by if c["status"]=="shortlisted"],
        "all": sorted_by,
        "invites_sent": invites_sent
    }), 200
sock = Sock(app)

@app.route("/api/upload_offer_template", methods=["POST"])
def upload_offer_template():
    employer_id = request.form.get("employer_id")
    file = request.files.get("file")

    if not file:
        return jsonify({"error": "No file uploaded"}), 400
    
    save_path = f"templates/{employer_id}_offer_template.docx"
    file.save(save_path)

    return jsonify({"msg": "Template uploaded", "path": save_path})


@app.route("/offer_accepted", methods=["POST"])
def offer_accepted():

    body = request.json

    candidate_id = body.get("candidate_id")
    candidate_name = body.get("candidate_name")
    job_role = body.get("job_role")
    job_type = body.get("job_type")
    company_name = body.get("company_name")
    job_description = body.get("job_description")
    training_months = body.get("training_months")
    company_type = body.get("company_type")
    employee_count = body.get("employee_count")

    # Save to DB
    data = Offeraccepted(
        candidate_id=candidate_id,
        candidate_name=candidate_name,
        job_role=job_role,
        job_type=job_type
   
    )
    db.session.add(data)
    db.session.commit()

    # Gemini Query Prompt
    prompt = f"""
    Suggest onboarding and job-specific training for a new hire:
    
    Candidate: {candidate_name}
    Role: {job_role}
    Company: {company_name}
    Company Type: {company_type}
    Employees: {employee_count}
    Job Description: {job_description}
    Training Duration (months): {training_months}

    Include:
    - Training Modules List
    - Company Culture & Policies Brief
    - Tools/Access Required
    - Progress & Assessment plan
    """

    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt)
    
    return jsonify({
        "message": "Offer accepted and onboarding plan generated successfully",
        "training_recommendation": response.text
    })

@app.route("/offer_accepted_candidates", methods=["GET"])
def get_offer_accepted_candidates():
    # Fetch all records from the Offeraccepted table
    candidates = Offeraccepted.query.all()
    
    # Convert the records to a list of dictionaries
    result = []
    for c in candidates:
        result.append({
            "candidate_id": c.candidate_id,
            "candidate_name": c.candidate_name,
            "job_role": c.job_role,
            "job_type": c.job_type
        })
    
    return jsonify({
        "message": "List of candidates who accepted offers",
        "candidates": result
    })


@app.route("/chat", methods=["POST"])
def employer_chat():
    body = request.get_json()

    candidate_id = body.get("candidate_id")
    message = body.get("message")

    if not candidate_id or not message:
        return jsonify({"error": "candidate_id and message required"}), 400

    # Save employer message as "user"
    db.session.add(ConversationHistory(
        candidate_id=candidate_id,
        role="user",
        content=message
    ))
    db.session.commit()

    # Fetch history
    history = ConversationHistory.query.filter_by(candidate_id=candidate_id).order_by(ConversationHistory.id.asc()).all()

    chat_context = ""
    for h in history:
        chat_context += f"{h.role}: {h.content}\n"

    prompt = f"""
    You are HR Onboarding Assistant helping an employer communicate with a new hire.

    Past Conversation:
    {chat_context}

    Latest Employer message: {message}

    Provide a friendly HR assistant response including:
    - onboarding guidance
    - next steps for the employee
    """

    model = genai.GenerativeModel("gemini-2.0-flash")

    response = model.generate_content(
        contents=[{
            "role": "user",
            "parts": [{"text": prompt}]
        }]
    )

    ai_reply_text = response.text

    # Save AI reply as "assistant"
    db.session.add(ConversationHistory(
        candidate_id=candidate_id,
        role="assistant",
        content=ai_reply_text
    ))
    db.session.commit()

    return jsonify({
        "reply": ai_reply_text,
        "candidate_id": candidate_id
    })

@app.route("/start-interview/<int:candidate_id>", methods=["GET"])
def start_interview(candidate_id):
    cand = Candidate.query.get(candidate_id)
    if not cand:
        return jsonify({"error":"Candidate not found"}), 404
    
    # Create an interview session in DB
    session = InterviewSession(candidate_id=candidate_id)
    db.session.add(session)
    db.session.commit()

    # Generate a simple session token
    token = f"session-{session.id}-{datetime.utcnow().timestamp()}"

    return jsonify({
        "session_id": session.id,
        "candidate_name": cand.name,
        "token": token,
        "ws_url": f"ws://localhost:8000/api/interview"  # Websocket endpoint
    })
@sock.route('/api/interview')
def realtime_interview(ws):
    """
    Real-time AI interview WebSocket:
    1. Receives candidate ID and session token
    2. Generates personalized questions
    3. Sends each question as audio/text
    4. Receives candidate answers
    5. Computes interview score at the end
    """
    try:
        # Receive initial payload with candidate_id
        init_payload = json.loads(ws.receive())
        candidate_id = init_payload.get("candidate_id")
        session_token = init_payload.get("token")

        cand = db.session.get(Candidate, candidate_id)
        if not cand:
            ws.send(json.dumps({"error": "Candidate not found"}))
            return

        session = InterviewSession(candidate_id=candidate_id)
        db.session.add(session)
        db.session.commit()

        # Generate personalized questions
        questions = generate_personalized_questions(cand.summary_text or "", init_payload.get("job_description",""))

        transcript = []

        ws.send(json.dumps({"msg": "Interview started.", "questions": questions}))
        
        #transcript_2 = [{"question": answer}, {"question": answer}]
        transcript_2 ={}
        # Iterate over questions%
        for q in questions:
            speak(q)
            ws.send(json.dumps({"question": q}))  # send question
            speech_answer = listen_until_silence()
            answer_payload = json.loads(ws.receive())
            ui_answer = answer_payload.get("answer", "")
            answer_text = speech_answer if speech_answer.strip() else ui_answer
            transcript.append(answer_text)
            transcript_2[q]=answer_text

        # Compute final interview score
        interview_score = score_interview(transcript, cand.summary_text or "", init_payload.get("job_description",""))

        # Save session details
        session.transcript = "\n".join(transcript)
        session.score = interview_score
        session.ended_at = datetime.utcnow()
        db.session.add(session)
        db.session.commit()

        ws.send(json.dumps({
            "msg": "Interview completed.",
            "transcript": transcript,
            "interview_score": interview_score
        }))


    except Exception as e:
        app.logger.exception("WebSocket interview error: %s", str(e))
        ws.send(json.dumps({"error": str(e)}))

@app.route("/api/generate_offer_letter", methods=["POST"])
def generate_offer_letter():
    """
    1. Receives candidate_id
    2. Checks interview score
    3. Generates offer letter from uploaded template
    4. Sends to candidate email
    """
    data = request.json
    candidate_id = data.get("candidate_id")

    cand = db.session.get(Candidate, candidate_id)
    print("candidate_id", candidate_id)
    print("cand", cand.name)
    if not cand:
        return jsonify({"error": "Candidate not found"}), 404

    session = (
        InterviewSession.query
        .filter_by(candidate_id=candidate_id)
        .order_by(InterviewSession.id.desc())
        .first()
    )

    if not session:
        print('4444')
        return jsonify({"error": "Interview session not found"}), 404

    if session.score < 3:
        print("5555")
        return jsonify({"msg": "Score below 90, no offer letter sent."}), 200


    # Template path
    template_path = os.path.join(app.root_path, "templates", f"1_offer_template.docx")
    print("Looking for template at:", template_path)
    if not os.path.exists(template_path):
        print('2222')
        return jsonify({"error": "Employer offer letter template missing"}), 404
    print('3333')
    # Placeholder context
    context = {
        "CANDIDATE_NAME": cand.name,
        "POSITION": cand.job_role or "AI/ML",
        "JOINING_DATE": "15-Dec-2025",
        "SALARY": cand.salary or 15000,
        "COMPANY_NAME": cand.company_name or "abc",
    }

    output_dir = "generated"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Generate DOCX
    output_docx = f"generated/offer_{candidate_id}.docx"
    doc = DocxTemplate(template_path)
    doc.render(context)
    doc.save(output_docx)

    pythoncom.CoInitialize()

    # Convert DOCX → PDF
    output_pdf = f"generated/offer_{candidate_id}.pdf"
    convert(output_docx, output_pdf)

    # Send email using new logging function
    ok, err = send_offer_letter_and_log(
        candidate_id=candidate_id,
        to_email=cand.email,
        candidate_name=cand.name,
        job_role=cand.job_role or "AI/ML",
        offer_file_path=output_pdf,
        company_name=cand.company_name,
    )

    if not ok:
        return jsonify({
            "msg": "Offer letter generated but email sending failed.",
            "error": err
        }), 500

    return jsonify({
        "msg": "Offer letter generated and sent successfully.",
        "file_path": output_pdf
    })
from flask import request, jsonify
from datetime import datetime, timedelta

@app.route("/create_internship_for_offer_accepted", methods=["POST"])
def create_internship_for_offer_accepted():
    """
    This endpoint takes a candidate_id and creates a new internship record
    with start and end dates for the candidate who accepted the offer.
    """
    data = request.get_json()
    candidate_id = data.get("candidate_id")
    internship_start_str = data.get("internship_start")  # format: "YYYY-MM-DD"
    internship_end_str = data.get("internship_end")      # format: "YYYY-MM-DD"

    if not candidate_id:
        return jsonify({"error": "candidate_id is required"}), 400

    # Fetch candidate details from Offeraccepted table
    candidate = Offeraccepted.query.filter_by(candidate_id=candidate_id).first()
    if not candidate:
        return jsonify({"error": "Candidate not found in offer accepted table"}), 404

    # Parse dates
    try:
        internship_start = datetime.strptime(internship_start_str, "%Y-%m-%d") if internship_start_str else datetime.today()
        internship_end = datetime.strptime(internship_end_str, "%Y-%m-%d") if internship_end_str else internship_start + timedelta(days=90)  # default 3 months
    except ValueError:
        return jsonify({"error": "Invalid date format. Use YYYY-MM-DD"}), 400

    # Create a new InternshipProcess record
    new_internship = InternshipProcess1(
        candidate_id=candidate.candidate_id,
        review_status="pending",  # default status
        internship_start=internship_start,
        internship_end=internship_end
    )

    # Save to DB
    db.session.add(new_internship)
    db.session.commit()

    return jsonify({
        "message": "Internship created successfully",
        "candidate_id": candidate.candidate_id,
        "internship_start": internship_start.strftime("%Y-%m-%d"),
        "internship_end": internship_end.strftime("%Y-%m-%d")
    })

@app.route("/templates/upload", methods=["POST"])
def upload_template():
    company = request.form.get("company_name")
    template = request.files.get("file")

    if not template or not company:
        return jsonify({"error": "Missing required fields"}), 400

    save_path = f"templates/{company}_{template.filename}"
    template.save(save_path)

    new = DocumentTemplate(
        template_name=template.filename,
        company_name=company,
        file_path=save_path
    )
    db.session.add(new)
    db.session.commit()

    return jsonify({"message": "Template uploaded successfully", "template_id": new.id})

@app.route("/check-internships", methods=["GET"])
def check_internships():
    all_internships = InternshipProcess1.query.all()  # get everything

    result = [
        {
            "candidate_id": i.candidate_id,
            "process_id": i.id,
            "review_status": i.review_status,
            "internship_start": i.internship_start.strftime("%Y-%m-%d") if i.internship_start else None,
            "internship_end": i.internship_end.strftime("%Y-%m-%d") if i.internship_end else None
        }
        for i in all_internships
    ]

    return jsonify(result)

@app.route("/certificate/approve", methods=["POST"])
def approve_certificate():
    data = request.json
    process_id = data.get("process_id")
    decision = data.get("decision")  # approved / rejected

    process = InternshipProcess.query.get(process_id)
    if not process:
        return jsonify({"error": "Invalid process"}), 404

    process.review_status = decision
    db.session.commit()

    if decision == "approved":
        return jsonify({
            "message": "Approved — generating certificate",
            "trigger_generate": True,
            "process_id": process.id
        })

    return jsonify({"message": "Rejected — no certificate"})

@app.route("/certificate/generate", methods=["POST"])
def generate_certificate():
    data = request.json
    process_ids = data.get("process_ids")
    template_id = data.get("template_id")

    template = DocumentTemplate.query.get(template_id)
    if not template:
        return jsonify({"error": "Invalid template"}), 400

    output_files = []

    for pid in process_ids:
        process = InternshipProcess1.query.get(pid)
        if not process:
            continue  # Skip invalid ID
        
        candidate = Candidate.query.get(process.candidate_id)
        doc = DocxTemplate(template.file_path)

        context = {
            "name": candidate.name,
            "email": candidate.email,
            "company": candidate.company_name,
            "role": candidate.job_role,
            "date": datetime.today().strftime("%d-%m-%Y")
        }
        doc.render(context)

        os.makedirs("generated_docs", exist_ok=True)
        output_path = f"generated_docs/{candidate.name}_certificate.docx"
        doc.save(output_path)

        process.approved_doc = output_path
        db.session.commit()

        output_files.append(output_path)

    return jsonify({
        "message": "Certificates generated successfully",
        "files": output_files
    })



if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    import logging
    logging.basicConfig(level=logging.INFO)
    app.run(host="0.0.0.0", port=8000, debug=True)
