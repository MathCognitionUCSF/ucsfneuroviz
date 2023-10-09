# ucsfneuroviz
Tools for an interactive browser page to visualize MRI data and behavioral scores in various contexts.

## Getting Started

### Clone the repository
git clone https://github.com/MathCognitionUCSF/ucsfneuroviz.git

### Navigate into Project
cd ucsfneuroviz

### Create a Virtual Environment
python3 -m venv venv (or python -m venv venv depending on your setup)

### Activate the Environment
Mac/Linux: source venv/bin/activate
Windows: venv\Scripts\activate

### Install Requirements
pip install -r requirements.txt

### Launch the browser page with Voila:
voila --Voila.config_file_paths=./.voila/ browser-page-filename.ipynb
