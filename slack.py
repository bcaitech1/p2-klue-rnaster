from functools import wraps
import requests
import traceback
from datetime import datetime
from pytz import timezone

from webhook_url import url


def alert(func):
    @wraps(func)
    def wrapper(*arg, **kwargs):
        start_time = datetime.now(timezone("asia/seoul"))
        text = "start! %s" % start_time.strftime("%Y-%m-%d %H:%M:%S")
        requests.post(url, json={"text": text})
        flag = True
        try:
            return func(*arg, **kwargs)
        except:
            requests.post(url, json={"text": traceback.format_exc()})
            flag = False
        finally:
            if flag:
                elapsed_time = datetime.now(timezone("asia/seoul")) - start_time
                text = "finish! elapsed time: %s" % elapsed_time
                requests.post(url, json={"text": text})
    return wrapper


@alert
def func_test():
    import time, random
    time.sleep(2)
    print("hello!")
    if random.random() > 0.1:
        return 1 / 0
    return


if __name__ == '__main__':
    func_test()
