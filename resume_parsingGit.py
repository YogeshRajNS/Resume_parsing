import os
import re
import tempfile
import traceback
from pathlib import Path
from typing import List, Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import json

from flask import Flask, request, jsonify, render_template
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import desc, func

# PDF/DOCX parsing
import pdfplumber
import docx

os.environ["GEMINI_API_KEY"] = "AIzaSyBLjy8o"
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
    "Summary": string or null
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
        app.logger.info("Gemini MODEL not configured — skipping LLM calls.")
        return ""
    try:
        resp = MODEL.generate_content(
            prompt_text,
            generation_config={"max_output_tokens": max_output_tokens}
        )
        if hasattr(resp, "text") and resp.text:
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
        app.logger.exception("Gemini call failed: %s", str(e))
        return ""

def gemini_extract(resume_text: str, job_description: str ="") -> dict:
    prompt_text = EXTRACTION_PROMPT_TEXT.format(resume_text=resume_text, job_description=job_description)
    out = call_gemini_text(prompt_text, max_output_tokens=3000)
    match = re.search(r'\$\$(.?)\$\$', out, re.DOTALL) 
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
SMTP_PASS = ""
FROM_EMAIL = SMTP_USER

def send_invite_email_and_log(candidate_id: Optional[int], to_email: str, candidate_name: str, job_title: str, scheduling_link: str="#", max_retries: int = 2):
    subject = f"Interview Invitation — {job_title}"
    body = f"Hi {candidate_name or ''},\n\nThanks for applying to {job_title}. Please schedule here: {scheduling_link}\n\nBest,\nRecruiting Team"
    log = EmailLog(candidate_id=candidate_id, to_email=to_email, subject=subject, body=body)
    db.session.add(log)
    db.session.commit()
    ok = False
    err = None
    for attempt in range(max_retries + 1):
        try:
            if not SMTP_HOST or not SMTP_USER or not SMTP_PASS:
                raise RuntimeError("SMTP credentials not configured in env (SMTP_HOST/SMTP_USER/SMTP_PASS)")
            msg = EmailMessage()
            msg["Subject"] = subject
            msg["From"] = FROM_EMAIL
            msg["To"] = to_email
            msg.set_content(body)
            with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=30) as s:
                s.starttls()
                s.login(SMTP_USER, SMTP_PASS)
                s.send_message(msg)
            ok = True
            break
        except Exception as e:
            err = str(e)
    log.sent_ok = bool(ok)
    log.error = err
    log.attempt_at = datetime.utcnow()
    db.session.add(log)
    db.session.commit()
    return ok, err

# ---------------- API / Processing ----------------

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
                status=status
            )
            db.session.add(cand)
            db.session.commit()

            if cand.score >= 70 and cand.email:
                send_invite_email_and_log(
                    candidate_id=cand.id,
                    to_email=cand.email,
                    candidate_name=cand.name,
                    job_title="Applied Role",
                    scheduling_link="https://calendly.com/your-link"
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
                    ok, err = send_invite_email_and_log(c.get("id"), c["email"], c.get("name"), job_title="Applied Role")
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


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    import logging
    logging.basicConfig(level=logging.INFO)
    app.run(host="0.0.0.0", port=port, debug=True)