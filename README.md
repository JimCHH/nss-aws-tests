`sudo nano /etc/rc.local`
```
#!/bin/bash

# DLAMI Configurations
/opt/aws/dlami/bin/init

# DLAMI Conda Configurations
/opt/aws/dlami/bin/init_conda

# auto.py
source /home/ubuntu/.bashrc
python /home/ubuntu/nss-aws-tests/auto.py

exit 0
```
