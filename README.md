`crontab -e`
```
SHELL=/bin/bash
BASH_ENV=~/.bashrc
#*/1 * * * * python3 ~/nss-aws-tests/auto.py
#*/1 * * * * source activate pytorch_p36 && python3 ~/nss-aws-tests/auto.py
@reboot sleep 90 ; source activate pytorch_p36 ; python3 ~/nss-aws-tests/auto.py
```
