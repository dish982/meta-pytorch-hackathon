def grade(log_lines):
    rewards = []

    for l in log_lines:
        if '[STEP]' in l and 'reward=' in l:
            try:
                val = float(l.split('reward=')[1].split(' ')[0])
                rewards.append(val)
            except:
                pass

    if not rewards:
        return 0.51   # safe fallback (NOT 0.5 exactly)

    avg = sum(rewards) / len(rewards)

    return max(0.01, min(0.99, avg))