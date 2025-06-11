# AI Web App with FastAPI and IBM AI

A modern web application built with FastAPI and IBM's generative AI models. This application provides a user-friendly interface for interacting with IBM's advanced AI capabilities.

## Features

- 🚀 FastAPI backend with async support
- 🎨 Modern UI with Tailwind CSS
- 🤖 IBM AI model integration
- 🌙 Dark mode support
- 📱 Responsive design
- 🔒 Secure environment variable handling

## Prerequisites

- Python 3.8 or higher
- IBM Cloud account with AI model access
- Git (optional, for version control)

## Setup

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd ai-web-app
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the root directory:
   ```env
   IBM_API_KEY=your_ibm_api_key_here
   IBM_PROJECT_ID=your_project_id_here
   IBM_ENDPOINT_URL=https://api.ibm.com/your-endpoint-url
   SECRET_KEY=your_secret_key_here
   DEBUG=True
   ENVIRONMENT=development
   ```

5. Start the development server:
   ```bash
   uvicorn main:app --reload
   ```

The application will be available at `http://localhost:8000`

## Project Structure

```
ai-web-app/
├── app/                    # FastAPI application code
│   ├── routers/           # API route handlers
│   └── models/            # Data models
├── templates/             # Jinja2 templates
│   ├── base.html         # Base template
│   ├── index.html        # Home page
│   └── ibm_model.html    # AI model interaction page
├── static/               # Static files
│   ├── css/             # Stylesheets
│   ├── js/              # JavaScript files
│   └── images/          # Image assets
├── main.py              # Application entry point
├── ibm_utils.py         # IBM AI integration utilities
├── requirements.txt     # Python dependencies
└── .env                 # Environment variables (create from .env.example)
```

## Usage

1. Visit the home page at `http://localhost:8000`
2. Navigate to the IBM AI Model page
3. Enter your text in the input field
4. Click "Process with AI" or press Ctrl+Enter to submit
5. View the AI model's response

## Development

- The application uses FastAPI for the backend
- Templates are written in Jinja2
- Styling is done with Tailwind CSS
- IBM AI integration is handled through the `ibm-generative-ai` package

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- FastAPI for the web framework
- IBM for the AI models
- Tailwind CSS for the styling
- Jinja2 for templating
