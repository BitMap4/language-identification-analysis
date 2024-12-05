#!/bin/bash
set -e

echo "Creating Python virtual environment..."
python3 -m venv venv

echo "Activating virtual environment..."
source venv/bin/activate

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing requirements..."
pip install -r requirements.txt

echo "Cloning resource repositories..."
mkdir indic_nlp_project
cd indic_nlp_project
git clone https://github.com/anoopkunchukuttan/indic_nlp_library.git
git clone https://github.com/anoopkunchukuttan/indic_nlp_resources.git
cd ..

echo "Downloading datasets..."
mkdir -p data/raw
wget https://huggingface.co/datasets/cfilt/IITB-IndicMonoDoc/resolve/main/mr/shard-12.txt
mv shard-12.txt data/raw/marathi.txt
wget https://huggingface.co/datasets/cfilt/IITB-IndicMonoDoc/resolve/main/hi/shard-30.txt
mv shard-30.txt data/raw/hindi.txt

echo "Making results directories..."
mkdir -p results/{2000,500}

echo "Setup completed successfully!"