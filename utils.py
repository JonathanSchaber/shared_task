import datetime
import re


def get_timestamp():
    timestamp = datetime.datetime.now().strftime('%c')
    return re.sub(r'\s+', r'_', timestamp)