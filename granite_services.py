import asyncio
import json
import logging
from typing import List, Dict, Any, Optional
from ibm_watson_machine_learning import APIClient
from sentence_transformers import SentenceTransformer
import numpy as np
from app.core.config import settings

logger = logging.getLogger(__name__)

class GraniteService:
    def __init__(self):
        self.client = None
        self.embedding_model = None
        self.is_ready = False
        
    async def initialize(self):
        """Initialize IBM Watson ML client and embedding model"""
        try:
            # Initialize IBM Watson ML client
            wml_credentials = {
                "url": settings.IBM_URL,
                "apikey": settings.IBM_API_KEY
            }
            
            self.client = APIClient(wml_credentials)
            
            if settings.IBM_SPACE_ID:
                self.client.set.default_space(settings.IBM_SPACE_ID)
            elif settings.IBM_PROJECT_ID:
                self.client.set.default_project(settings.IBM_PROJECT_ID)
            
            # Initialize sentence transformer for embeddings
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            self.is_ready = True
            logger.info("Granite service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Granite service: {str(e)}")
            self.is_ready = False
    
    async def cleanup(self):
        """Cleanup resources"""
        self.is_ready = False
        logger.info("Granite service cleaned up")
    
    async def generate_store_job_description(self, store_job_input: Dict[str, Any]) -> Dict[str, str]:
        """
        Generate a professional job description for store owners using IBM Granite 3.3
        """
        try:
            # Create a structured prompt for store job posting
            prompt = f"""
            You are a professional job posting writer. Create a clear, complete, and industry-appropriate job description for a local store/business based on the following information:

            Job Title: {store_job_input.get('job_title', '')}
            Store/Business: {store_job_input.get('store_name', '')}
            Location: {store_job_input.get('location', '')}
            Key Responsibilities: {store_job_input.get('key_responsibilities', 'Not specified')}
            Skills Required: {store_job_input.get('skills_required', 'Not specified')}
            Working Hours: {store_job_input.get('working_hours', 'Not specified')}
            Working Days: {store_job_input.get('working_days', 'Not specified')}
            Salary: {store_job_input.get('salary', 'Competitive salary')}
            Job Type: {store_job_input.get('job_type', 'Full-time')}
            Contact Info: {store_job_input.get('contact_info', 'Apply in person')}
            Additional Info: {store_job_input.get('additional_info', '')}

            Create a professional job posting that includes:
            1. A clear job title and company name
            2. Job overview and main responsibilities
            3. Required skills and qualifications
            4. Working hours and schedule
            5. Salary information
            6. Location details
            7. How to apply

            Format it as a complete, ready-to-post job description that would attract suitable candidates for a local store position.
            Make it professional but accessible, suitable for local job seekers.
            """
            
            # Generate the job description
            enhanced_description = await self._generate_text(prompt)
            
            # Generate a summary
            summary_prompt = f"""
            Create a brief, engaging summary (1-2 sentences) for this job posting:
            
            {enhanced_description}
            
            The summary should highlight the key role, location, and main appeal to job seekers.
            Keep it under 100 words and make it attractive for mobile job browsing.
            """
            
            summary = await self._generate_text(summary_prompt)
            
            # Create a formatted job post
            formatted_post = self._format_job_post(store_job_input, enhanced_description)
            
            return {
                'enhanced_description': enhanced_description,
                'summary': summary,
                'formatted_post': formatted_post
            }
            
        except Exception as e:
            logger.error(f"Error generating store job description: {str(e)}")
            return {
                'enhanced_description': self._create_fallback_description(store_job_input),
                'summary': f"{store_job_input.get('job_title', 'Job')} position available at {store_job_input.get('store_name', 'local store')}",
                'formatted_post': self._create_fallback_description(store_job_input)
            }
    
    def _format_job_post(self, input_data: Dict[str, Any], ai_description: str) -> str:
        """Format the job post in a structured, mobile-friendly way"""
        
        formatted_post = f"""
🏪 **{input_data.get('job_title', 'Job Position')}**
📍 {input_data.get('store_name', 'Local Store')} - {input_data.get('location', 'Location TBD')}

{ai_description}

📋 **Quick Details:**
• Working Hours: {input_data.get('working_hours', 'To be discussed')}
• Working Days: {input_data.get('working_days', 'To be discussed')}
• Salary: {input_data.get('salary', 'Competitive')}
• Job Type: {input_data.get('job_type', 'Full-time')}

📞 **How to Apply:**
{input_data.get('contact_info', 'Apply in person at the store location')}
        """.strip()
        
        return formatted_post
    
    def _create_fallback_description(self, input_data: Dict[str, Any]) -> str:
        """Create a basic job description when AI is unavailable"""
        
        description = f"""
**{input_data.get('job_title', 'Job Position')} - {input_data.get('store_name', 'Local Store')}**

We are looking for a {input_data.get('job_title', 'team member')} to join our team at {input_data.get('store_name', 'our store')}.

**Location:** {input_data.get('location', 'Please inquire')}

**Responsibilities:**
{input_data.get('key_responsibilities', 'Various store duties as assigned')}

**Requirements:**
{input_data.get('skills_required', 'Willingness to learn and work as part of a team')}

**Schedule:**
• Hours: {input_data.get('working_hours', 'To be discussed')}
• Days: {input_data.get('working_days', 'To be discussed')}

**Compensation:** {input_data.get('salary', 'Competitive salary')}

**How to Apply:** {input_data.get('contact_info', 'Apply in person')}
        """.strip()
        
        return description
    
    async def generate_smart_job_description(self, job_data: Dict[str, Any]) -> str:
        """
        Generate an enhanced job description using IBM Granite 3.3
        """
        try:
            prompt = f"""
            Create a comprehensive and engaging job description based on the following information:
            
            Job Title: {job_data.get('title', '')}
            Company: {job_data.get('company', '')}
            Location: {job_data.get('location', '')}
            Basic Description: {job_data.get('description', '')}
            Requirements: {job_data.get('requirements', '')}
            Experience Level: {job_data.get('experience_level', '')}
            Job Type: {job_data.get('job_type', '')}
            
            Please create a professional, detailed job description that includes:
            1. An engaging overview of the role
            2. Key responsibilities
            3. Required qualifications
            4. Preferred qualifications
            5. What the company offers
            6. Growth opportunities
            
            Make it attractive to potential candidates while being clear about expectations.
            """
            
            # Generate using IBM Granite
            response = await self._generate_text(prompt)
            return response
            
        except Exception as e:
            logger.error(f"Error generating smart job description: {str(e)}")
            return job_data.get('description', '')
    
    async def generate_job_summary(self, job_description: str) -> str:
        """
        Generate an AI-powered job summary using IBM Granite 3.3
        """
        try:
            prompt = f"""
            Create a concise, engaging summary (2-3 sentences) of the following job description:
            
            {job_description}
            
            The summary should:
            1. Highlight the key role and main responsibilities
            2. Mention the most important qualifications
            3. Be appealing to job seekers
            4. Be under 150 words
            """
            
            response = await self._generate_text(prompt)
            return response
            
        except Exception as e:
            logger.error(f"Error generating job summary: {str(e)}")
            return job_description[:200] + "..."
    
    async def calculate_job_match_score(self, user_profile: Dict[str, Any], job_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate job matching score using AI analysis
        """
        try:
            # Create embeddings for user profile and job
            user_text = self._create_user_profile_text(user_profile)
            job_text = self._create_job_text(job_data)
            
            user_embedding = self.embedding_model.encode([user_text])
            job_embedding = self.embedding_model.encode([job_text])
            
            # Calculate cosine similarity
            similarity = np.dot(user_embedding[0], job_embedding[0]) / (
                np.linalg.norm(user_embedding[0]) * np.linalg.norm(job_embedding[0])
            )
            
            # Convert to percentage
            base_score = float(similarity) * 100
            
            # Use Granite for detailed analysis
            analysis_prompt = f"""
            Analyze the job match between this candidate and job posting:
            
            CANDIDATE PROFILE:
            {user_text}
            
            JOB POSTING:
            {job_text}
            
            Provide a detailed analysis including:
            1. Overall match percentage (0-100)
            2. Strengths (what matches well)
            3. Gaps (what's missing)
            4. Recommendations for the candidate
            
            Format as JSON with keys: match_score, strengths, gaps, recommendations
            """
            
            analysis = await self._generate_text(analysis_prompt)
            
            try:
                # Try to parse as JSON
                detailed_analysis = json.loads(analysis)
                match_score = detailed_analysis.get('match_score', base_score)
            except:
                # Fallback to base score if JSON parsing fails
                match_score = base_score
                detailed_analysis = {
                    'match_score': match_score,
                    'strengths': ['Profile matches job requirements'],
                    'gaps': ['Some requirements may not be fully met'],
                    'recommendations': ['Review job requirements and highlight relevant experience']
                }
            
            return {
                'match_score': min(100, max(0, match_score)),
                'analysis': detailed_analysis
            }
            
        except Exception as e:
            logger.error(f"Error calculating job match score: {str(e)}")
            return {
                'match_score': 50.0,
                'analysis': {
                    'match_score': 50.0,
                    'strengths': ['Unable to analyze at this time'],
                    'gaps': ['Analysis unavailable'],
                    'recommendations': ['Please try again later']
                }
            }
    
    async def _generate_text(self, prompt: str) -> str:
        """
        Generate text using IBM Granite model
        """
        try:
            # Prepare the generation parameters
            generation_params = {
                "max_new_tokens": 800,
                "temperature": 0.7,
                "top_p": 0.9,
                "repetition_penalty": 1.1
            }
            
            # Generate text using the model
            if settings.GRANITE_DEPLOYMENT_ID:
                # Use deployment if available
                response = self.client.deployments.generate_text(
                    deployment_id=settings.GRANITE_DEPLOYMENT_ID,
                    prompt=prompt,
                    params=generation_params
                )
            else:
                # Use foundation model directly
                response = self.client.foundation_models.generate_text(
                    model_id=settings.GRANITE_MODEL_ID,
                    prompt=prompt,
                    params=generation_params
                )
            
            return response.get('results', [{}])[0].get('generated_text', '').strip()
            
        except Exception as e:
            logger.error(f"Error generating text with Granite: {str(e)}")
            # Fallback to a simple response
            return "AI generation temporarily unavailable. Please try again later."
    
    def _create_user_profile_text(self, user_profile: Dict[str, Any]) -> str:
        """Create a text representation of user profile for embedding"""
        parts = []
        
        if user_profile.get('skills'):
            parts.append(f"Skills: {user_profile['skills']}")
        
        if user_profile.get('experience_years'):
            parts.append(f"Experience: {user_profile['experience_years']} years")
        
        if user_profile.get('education'):
            parts.append(f"Education: {user_profile['education']}")
        
        if user_profile.get('bio'):
            parts.append(f"Bio: {user_profile['bio']}")
        
        if user_profile.get('location'):
            parts.append(f"Location: {user_profile['location']}")
        
        return " | ".join(parts)
    
    def _create_job_text(self, job_data: Dict[str, Any]) -> str:
        """Create a text representation of job for embedding"""
        parts = []
        
        if job_data.get('title'):
            parts.append(f"Title: {job_data['title']}")
        
        if job_data.get('description'):
            parts.append(f"Description: {job_data['description']}")
        
        if job_data.get('requirements'):
            parts.append(f"Requirements: {job_data['requirements']}")
        
        if job_data.get('skills_required'):
            parts.append(f"Skills: {job_data['skills_required']}")
        
        if job_data.get('experience_level'):
            parts.append(f"Experience Level: {job_data['experience_level']}")
        
        if job_data.get('location'):
            parts.append(f"Location: {job_data['location']}")
        
        return " | ".join(parts)

# Global instance
granite_service = GraniteService()
