conda create -n multiview python=3.12
conda init
conda activate multiview
pip install -r requirements.txt
export $(cat .env | xargs)