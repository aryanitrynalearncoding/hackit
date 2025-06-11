import os
from ibm_generative_ai import GenerativeModel, Credentials

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def get_ibm_credentials():
    """Get IBM credentials from environment variables."""
    api_key = os.getenv("IBM_API_KEY")
    project_id = os.getenv("IBM_PROJECT_ID")
    
    if not all([api_key, project_id]):
        raise ValueError("Missing required IBM credentials in environment variables")
    
    return Credentials(api_key=api_key, project_id=project_id)

async def generate_job_description(role: str, skills: str, hours: str) -> str:
    """
    Generate a job description using IBM's Granite-13b-instruct model.
    
    Args:
        role (str): The job role/title
        skills (str): Required skills for the position
        hours (str): Working hours
        
    Returns:
        str: Generated job description
        
    Raises:
        Exception: If there's an error with the IBM API call
    """
    try:
        # Initialize the model with Granite-13b-instruct
        credentials = get_ibm_credentials()
        model = GenerativeModel(
            model_id="ibm/granite-13b-instruct-v1",  # Granite-13b-instruct model
            credentials=credentials
        )
        
        # Construct the prompt
        prompt = f"""Write a detailed job description for a {role} position at a local store.
Required skills: {skills}
Working hours: {hours}

Please include:
1. A compelling job overview
2. Detailed responsibilities
3. Required qualifications and skills
4. Working conditions and benefits
5. How to apply

Format the response in clear sections with bullet points where appropriate."""

        # Generate response with appropriate parameters for job description
        response = model.generate(
            prompt=prompt,
            max_new_tokens=1000,  # Longer response for detailed job description
            temperature=0.7,      # Balanced between creativity and consistency
            top_p=0.9,           # Allow some diversity in responses
            repetition_penalty=1.1 # Slightly penalize repetition
        )
        
        return response.generated_text
        
    except Exception as e:
        raise Exception(f"Error generating job description: {str(e)}")

def format_job_description(text: str) -> str:
    """
    Format the generated job description for HTML display.
    Converts newlines to <br> tags and ensures proper spacing.
    """
    # Split into sections and format
    sections = text.split('\n\n')
    formatted_sections = []
    
    for section in sections:
        if section.strip():
            # Convert bullet points to HTML list items
            if section.strip().startswith('•') or section.strip().startswith('-'):
                items = [item.strip('•- ').strip() for item in section.split('\n') if item.strip()]
                formatted_section = '<ul class="list-disc pl-6 mb-4">\n'
                formatted_section += '\n'.join(f'<li>{item}</li>' for item in items)
                formatted_section += '\n</ul>'
            else:
                # Regular paragraph
                formatted_section = f'<p class="mb-4">{section.strip()}</p>'
            formatted_sections.append(formatted_section)
    
    return '\n'.join(formatted_sections) 