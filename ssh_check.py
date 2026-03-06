import paramiko

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('connect.westd.seetacloud.com', port=14918, username='root', password='UvUnT2x1jsaa', timeout=15)

# Check training status
stdin, stdout, stderr = client.exec_command('tail -30 /root/autodl-tmp/arm_grasp/train_pick_place.log 2>/dev/null || tail -30 /root/autodl-tmp/arm_grasp/train.log 2>/dev/null', timeout=15)
print('=== Training Log ===')
print(stdout.read().decode())

stdin, stdout, stderr = client.exec_command('ls -la /root/autodl-tmp/arm_grasp/logs/rsl_rl/arm_pick_place_v9_5/model_*.pt 2>/dev/null | tail -5', timeout=10)
print('=== Checkpoints ===')
print(stdout.read().decode())

stdin, stdout, stderr = client.exec_command('ps aux | grep train | grep python | grep -v grep', timeout=5)
print('=== Processes ===')
print(stdout.read().decode())

client.close()
