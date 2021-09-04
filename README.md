1. `bash setup.sh`
2. `sudo cp -r .aws /root`
3. `sudo nano /root/.bashrc`
```
. /home/ubuntu/.bashrc
```
4. `sudo nano /etc/rc.local`
```
python /home/ubuntu/nss-aws-tests/auto.py
```
