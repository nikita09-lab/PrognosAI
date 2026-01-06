CRITICAL = 10
WARNING = 30

def health_status(rul):
    if rul <= CRITICAL:
        return "CRITICAL ⚠️"
    elif rul <= WARNING:
        return "WARNING ⚠"
    return "NORMAL ✓"
