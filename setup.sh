sudo apt update
sudo apt install -y automake autotools-dev fuse g++ git libcurl4-openssl-dev libfuse-dev libssl-dev libxml2-dev make pkg-config
sudo apt install -y s3fs
sudo apt install -y chromium-chromedriver --fix-missing

source activate pytorch_p36
pip install --upgrade pip
pip install --upgrade torch
pip install --upgrade xmltodict
pip install varname
pip install reportlab
pip install svglib
pip install selenium

echo source activate pytorch_p36 >> ~/.bashrc
echo sudo umount S3 >> ~/.bashrc
echo s3fs neurobit-asg ~/S3 -o passwd_file=~/.aws/credentials-s3fs -o uid=1000 -o gid=1000 >> ~/.bashrc

touch auto.py.log
