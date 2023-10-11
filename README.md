# ucsfneuroviz
Tools for an interactive browser page to visualize MRI data and behavioral scores in various contexts.

## Getting Started

## Setup:

#### 1. Clone the repository
git clone https://github.com/MathCognitionUCSF/ucsfneuroviz.git

#### 2. Navigate into Repository 
cd ucsfneuroviz

#### 3. Create a Virtual Environment
python3 -m venv ucsfneuroviz (or python -m venv venv depending on your setup)

#### 4. Activate the Environment
Mac/Linux: source venv/bin/activate
Windows: venv\Scripts\activate

#### 5. Install Requirements
pip install -r requirements.txt


## How to use:

#### 1. Navigate into Repository
cd ucsfneuroviz

#### 2. Activate the Environment
Mac/Linux: source venv/bin/activate &nbsp
Windows: venv\Scripts\activate

#### 3. Connect to L-drive
connect to the L-drive server so your local computer has access to the necessary data

#### 4. Launch the browser page with Voila
voila --Voila.config_file_paths=./.voila/ dyslexia-browser.ipynb
