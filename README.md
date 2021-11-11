# Automation setup from ubuntu to root
0. `git clone https://github.com/x1001000/nss-aws-tests`
1. `bash setup.sh`
2. `sudo cp -r .aws /root`
3. `sudo nano /etc/rc.local`
```
source /home/ubuntu/anaconda3/condabin/activate /home/ubuntu/anaconda3/envs/pytorch_p36
cd /home/ubuntu/nss-aws-tests/
git pull
python auto.py
```
4. `sudo systemctl status rc-local.service`
