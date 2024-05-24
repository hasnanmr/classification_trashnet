# Deep Learning Image Classification with PyTorch

This repository contains code for a deep learning image classification project using PyTorch. The project utilizes a dataset from Hugging Face and includes a CI/CD pipeline configured with GitHub Actions.

## Project Structure



## Project Structure

.
├── .github
│ └── workflows
│ └── ci-cd-pipeline.yml # GitHub Actions workflow file
├── download_dataset.py # Script to download dataset from Hugging Face
├── requirements.txt # Python dependencies
└── train.py # Script to train the deep learning model


## Setup Instructions

To set up the project on your local machine, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/your-repo-name.git
   cd your-repo-name/path

2. python -m venv venv
* on ubuntu or mac* 
'source venv/bin/activate'
*On Windows use*
 `venv\Scripts\activate`

3. python -m pip install --upgrade pip
pip install -r requirements.txt

4. python train.py
