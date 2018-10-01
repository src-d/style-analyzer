import cProfile
from functools import wraps
import pstats
import io


def profile(func):
    @wraps(func)
    def wrapped_profile(*args, **kwargs):
        pr = cProfile.Profile()
        pr.enable()
        try:
            res = func(*args, **kwargs)
        finally:
            pr.disable()
            s = io.StringIO()
            ps = pstats.Stats(pr, stream=s).strip_dirs().sort_stats("cumulative", "time")
            ps.print_stats(20)
            print(s.getvalue())
        return res
    return wrapped_profile
