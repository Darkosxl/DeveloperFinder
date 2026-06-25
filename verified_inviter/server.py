from verified_inviter import config
from verified_inviter.dashboard import app
from verified_inviter.scheduler import SCHEDULER

if config.SCHEDULER_AUTOSTART:
    SCHEDULER.start()
