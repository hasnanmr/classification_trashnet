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

## CI/CD

Description of the CI/CD pipeline and how it works:

- **GitHub Actions**: Automated workflows defined in `.github/workflows/ci-cd-pipeline.yml`.

## License

Include license information here.
MIT License

Copyright (c) [year] [fullname]

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
