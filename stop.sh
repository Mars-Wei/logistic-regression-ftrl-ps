ps -ef | grep ps_lr_ftrl | awk '{ print $2 }' | sudo xargs kill -9
