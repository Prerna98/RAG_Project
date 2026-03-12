import chromadb
from sentence_transformers import SentenceTransformer
from rapidfuzz import fuzz

model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_collection("resume_collection")

# -------------------------
# Extract Skills from JD
# -------------------------
def extract_skills(text):

    skills = [
        "Python", "Machine Learning", "SQL",
        "Deep Learning", "NLP", "Java", "AWS"
    ]

    found = []

    for s in skills:
        if s.lower() in text.lower():
            found.append(s)

    return found


# -------------------------
# Must Have Requirement
# -------------------------
def extract_experience_requirement(text):

    import re

    match = re.search(r"(\d+)\+?\s+years", text)

    if match:
        return int(match.group(1))

    return 0


# -------------------------
# Match Scoring
# -------------------------
def compute_match_score(resume_text, jd_text):

    semantic_score = fuzz.token_set_ratio(resume_text, jd_text) / 100

    return semantic_score * 100


# -------------------------
# Job Matching
# -------------------------
def match_job(job_description):

    jd_embedding = model.encode(job_description).tolist()

    results = collection.query(
        query_embeddings=[jd_embedding],
        n_results=10
    )

    required_skills = extract_skills(job_description)
    required_exp = extract_experience_requirement(job_description)

    matches = []

    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):

        if meta["experience_years"] < required_exp:
            continue

        score = compute_match_score(doc, job_description)

        matched_skills = [
            s for s in required_skills
            if s.lower() in doc.lower()
        ]

        matches.append({
            "candidate_name": meta["name"],
            "resume_path": meta["resume_path"],
            "match_score": round(score),
            "matched_skills": matched_skills,
            "relevant_excerpts": [doc[:200]],
            "reasoning": f"Matched skills: {matched_skills}"
        })

    matches = sorted(matches, key=lambda x: x["match_score"], reverse=True)

    return {
        "job_description": job_description,
        "top_matches": matches[:10]
    }


if __name__ == "__main__":

    jd = """
    Looking for a Machine Learning Engineer with 5+ years experience.
    Must have Python, Deep Learning, and AWS experience.
    """

    result = match_job(jd)

    from pprint import pprint
    pprint(result)