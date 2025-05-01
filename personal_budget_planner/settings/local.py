from .base import *
import socket

DEBUG = True

ALLOWED_HOSTS = ["127.0.0.1", "localhost"]
CSRF_TRUSTED_ORIGINS = [
    "http://127.0.0.1:8080",
    "http://localhost:8080"
]

if DEBUG:
    INSTALLED_APPS += [
        "debug_toolbar",
    ]

    MIDDLEWARE.insert(
        0,
        "debug_toolbar.middleware.DebugToolbarMiddleware",
    )


def get_internal_ips() -> list[str]:
    """
    Get internal IPs required for Django Debug Toolbar in a Docker environment.
    """
    hostname, _, ips = socket.gethostbyname_ex(socket.gethostname())
    internal_ips = [ip[: ip.rfind(".")] + ".1" for ip in ips] + ["127.0.0.1", "10.0.2.2"]

    # Since our requests will be routed to Django via the nginx container, include
    # the nginx IP address as internal as well
    try:
        nginx_hostname, _, nginx_ips = socket.gethostbyname_ex("nginx")
        internal_ips += nginx_ips
    except socket.gaierror:
        # since nginx may not be started at the point that this is first executed.
        pass
    return internal_ips


INTERNAL_IPS = get_internal_ips()

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.postgresql",
        "NAME": os.environ.get("DB_NAME", "postgres"),
        "USER": os.environ.get("DB_USER", "postgres"),
        "PASSWORD": os.environ.get("DB_PASSWORD", "postgres"),
        "HOST": os.environ.get("DB_HOST", "db"),
        "PORT": os.environ.get("DB_PORT", "5432"),
    }
}
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

CELERY_BROKER_URL = 'redis://redis:6379/0'
CELERY_RESULT_BACKEND = 'redis://redis:6379/0'
