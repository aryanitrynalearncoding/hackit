from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pathlib import Path
import os
from dotenv import load_dotenv
from ibm_utils import generate_job_description, format_job_description

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="Smart Job Description Generator")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Setup templates
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Render the home page."""
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "title": "Smart Job Description Generator"}
    )

@app.get("/generate-job-description", response_class=HTMLResponse)
async def job_description_form(request: Request):
    """Render the job description generation form."""
    return templates.TemplateResponse(
        "generate_job.html",
        {
            "request": request,
            "title": "Generate Job Description",
            "error": None,
            "job_description": None
        }
    )

@app.post("/generate-job-description", response_class=HTMLResponse)
async def generate_job_description_post(
    request: Request,
    role: str = Form(...),
    skills: str = Form(...),
    hours: str = Form(...)
):
    """Process the job description generation form."""
    try:
        # Generate job description using IBM Granite model
        description = await generate_job_description(role, skills, hours)
        
        # Format the description for HTML display
        formatted_description = format_job_description(description)
        
        return templates.TemplateResponse(
            "generate_job.html",
            {
                "request": request,
                "title": "Generated Job Description",
                "role": role,
                "skills": skills,
                "hours": hours,
                "job_description": formatted_description,
                "error": None
            }
        )
    except Exception as e:
        return templates.TemplateResponse(
            "generate_job.html",
            {
                "request": request,
                "title": "Generate Job Description",
                "role": role,
                "skills": skills,
                "hours": hours,
                "error": str(e),
                "job_description": None
            }
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 