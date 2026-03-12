import os
import re
from pypdf import PdfReader
import chromadb
from sentence_transformers import SentenceTransformer

# embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_or_create_collection(name="resume_collection")

# -------------------------
# Resume Loader
# -------------------------
def load_resume_text(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text


# -------------------------
# Section Aware Chunking
# -------------------------
def chunk_resume(text):

    sections = re.split(
        r"(Education|Experience|Skills|Projects|Certifications)",
        text,
        flags=re.IGNORECASE,
    )

    chunks = []

    for i in range(1, len(sections), 2):
        section = sections[i]
        content = sections[i + 1]

        # split long sections
        subchunks = [content[j:j + 500] for j in range(0, len(content), 500)]

        for c in subchunks:
            chunks.append({
                "section": section,
                "text": c
            })

    return chunks


# -------------------------
# Metadata Extraction
# -------------------------
def extract_metadata(text):

    name = text.split("\n")[0]

    skills = re.findall(
        r"(Python|Machine Learning|SQL|Deep Learning|NLP|Java|AWS)",
        text,
        re.IGNORECASE
    )

    exp_match = re.search(r"(\d+)\+?\s+years", text, re.IGNORECASE)
    experience = int(exp_match.group(1)) if exp_match else 0

    education = "Unknown"
    if "bachelor" in text.lower():
        education = "Bachelor"
    elif "master" in text.lower():
        education = "Master"

    return {
        "name": name,
        "skills": list(set(skills)),
        "experience_years": experience,
        "education": education
    }


# -------------------------
# Store Resumes
# -------------------------
def index_resumes(resume_folder):

    for file in os.listdir(resume_folder):

        if not file.endswith(".pdf"):
            continue

        path = os.path.join(resume_folder, file)

        text = load_resume_text(path)

        metadata = extract_metadata(text)

        chunks = chunk_resume(text)

        for i, chunk in enumerate(chunks):

            embedding = model.encode(chunk["text"]).tolist()

            collection.add(
                embeddings=[embedding],
                documents=[chunk["text"]],
                metadatas=[{
                    **metadata,
                    "section": chunk["section"],
                    "resume_path": path
                }],
                ids=[f"{file}_{i}"]
            )


if __name__ == "__main__":

    resume_folder = "resumes"

    index_resumes(resume_folder)


    print("Resumes indexed successfully")