1. `bash setup.sh`
1. `sudo cp -r .aws /root`
1. `sudo nano /etc/rc.local`
```
#!/bin/bash

# DLAMI Configurations
/opt/aws/dlami/bin/init

# DLAMI Conda Configurations
/opt/aws/dlami/bin/init_conda

# auto.py
python /home/ubuntu/nss-aws-tests/auto.py

exit 0
```
