1. `bash setup.sh`
2. `sudo cp -r .aws /root`
3. `sudo nano /etc/rc.local`
```
#. /home/ubuntu/.bashrc && python /home/ubuntu/nss-aws-tests/auto.py
source /home/ubuntu/anaconda3/condabin/activate /home/ubuntu/anaconda3/envs/pytorch_p36
python /home/ubuntu/nss-aws-tests/auto.py
```
4. `sudo systemctl status rc-local.service`
