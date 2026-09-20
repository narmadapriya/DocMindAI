"""SMTP delivery for DocMindAI password-recovery links.

Recovery configuration is deliberately kept outside the frozen application
Settings model and database schema. Values can be provided by the process
environment, ``backend/.password_reset.env`` (legacy/local override), or the
project's existing ``backend/.env`` file.
"""

from email.message import EmailMessage
import os
from pathlib import Path
import smtplib
import ssl
from urllib.parse import urlencode


class PasswordResetEmailConfigurationError(RuntimeError):
    """Raised when password-recovery SMTP settings are missing or invalid."""


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_RECOVERY_ENV_FILE = _PROJECT_ROOT / ".password_reset.env"
_PROJECT_ENV_FILE = _PROJECT_ROOT / ".env"


def _read_env_file(path: Path) -> dict[str, str]:
    """Read simple KEY=VALUE settings without mutating ``os.environ``."""
    if not path.is_file():
        return {}

    values: dict[str, str] = {}

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()

        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")

        if key:
            values[key] = value

    return values


def _local_recovery_settings() -> dict[str, str]:
    """Merge standard project env values with an optional recovery override."""
    values = _read_env_file(_PROJECT_ENV_FILE)
    values.update(_read_env_file(_RECOVERY_ENV_FILE))
    return values


def _env(name: str, default: str = "") -> str:
    """Resolve a recovery setting with process environment taking precedence."""
    process_value = os.getenv(name)
    if process_value is not None:
        return str(process_value).strip()

    file_value = _local_recovery_settings().get(name)
    if file_value is not None:
        return str(file_value).strip()

    return str(default or "").strip()


def _env_bool(name: str, default: bool) -> bool:
    raw = _env(name, "true" if default else "false").lower()
    return raw in {"1", "true", "yes", "on"}


def _looks_like_placeholder(value: str) -> bool:
    normalized = str(value or "").strip().lower()

    if not normalized:
        return False

    placeholder_fragments = (
        "your-email",
        "your_app",
        "your-app",
        "example.com",
        "changeme",
        "replace-me",
        "replace_me",
    )

    return any(fragment in normalized for fragment in placeholder_fragments)


def password_reset_email_configuration_error() -> str | None:
    """Return a safe configuration error message, or ``None`` when usable."""
    host = _env("DOCMINDAI_SMTP_HOST")
    username = _env("DOCMINDAI_SMTP_USERNAME")
    password = _env("DOCMINDAI_SMTP_PASSWORD")
    from_email = _env("DOCMINDAI_SMTP_FROM_EMAIL") or username

    missing: list[str] = []

    if not host or _looks_like_placeholder(host):
        missing.append("DOCMINDAI_SMTP_HOST")

    if not from_email or _looks_like_placeholder(from_email):
        missing.append("DOCMINDAI_SMTP_FROM_EMAIL")

    if bool(username) != bool(password):
        missing.append("SMTP username/password pair")
    elif username and (
        _looks_like_placeholder(username) or _looks_like_placeholder(password)
    ):
        missing.append("SMTP username/password pair")

    if missing:
        return (
            "Password recovery email is not configured on the server. "
            "Configure the DOCMINDAI_SMTP_* values in backend/.env and restart "
            "the backend."
        )

    try:
        port = int(_env("DOCMINDAI_SMTP_PORT", "587"))
    except ValueError:
        return "DOCMINDAI_SMTP_PORT must be a valid integer."

    if not 1 <= port <= 65535:
        return "DOCMINDAI_SMTP_PORT must be between 1 and 65535."

    if _env_bool("DOCMINDAI_SMTP_USE_SSL", False) and _env_bool(
        "DOCMINDAI_SMTP_USE_TLS",
        False,
    ):
        return "Enable either SMTP TLS or SMTP SSL, not both."

    return None


def password_reset_email_is_configured() -> bool:
    return password_reset_email_configuration_error() is None


def send_password_reset_email(
    recipient: str,
    token: str,
) -> None:
    """Send a short-lived password-reset link without exposing its token via API."""
    configuration_error = password_reset_email_configuration_error()
    if configuration_error:
        raise PasswordResetEmailConfigurationError(configuration_error)

    host = _env("DOCMINDAI_SMTP_HOST")
    username = _env("DOCMINDAI_SMTP_USERNAME")
    password = _env("DOCMINDAI_SMTP_PASSWORD")
    from_email = _env("DOCMINDAI_SMTP_FROM_EMAIL") or username
    use_ssl = _env_bool("DOCMINDAI_SMTP_USE_SSL", False)
    use_tls = _env_bool("DOCMINDAI_SMTP_USE_TLS", not use_ssl)

    default_port = "465" if use_ssl else "587"
    try:
        port = int(_env("DOCMINDAI_SMTP_PORT", default_port))
    except ValueError as exc:
        raise PasswordResetEmailConfigurationError(
            "DOCMINDAI_SMTP_PORT must be a valid integer."
        ) from exc

    frontend_url = _env(
        "DOCMINDAI_FRONTEND_URL",
        "http://localhost:5173",
    ).rstrip("/")

    reset_link = (
        f"{frontend_url}/forgot-password?"
        f"{urlencode({'token': token})}"
    )

    message = EmailMessage()
    message["Subject"] = "Reset your DocMind AI password"
    message["From"] = from_email
    message["To"] = recipient
    message.set_content(
        "A password reset was requested for your DocMind AI account.\n\n"
        f"Open this link to choose a new password:\n{reset_link}\n\n"
        "This link expires shortly and becomes invalid immediately after your "
        "password is changed. If you did not request this reset, you can ignore "
        "this email."
    )

    tls_context = ssl.create_default_context()

    if use_ssl:
        smtp = smtplib.SMTP_SSL(
            host,
            port,
            timeout=15,
            context=tls_context,
        )
    else:
        smtp = smtplib.SMTP(host, port, timeout=15)

    with smtp:
        smtp.ehlo()

        if use_tls:
            smtp.starttls(context=tls_context)
            smtp.ehlo()

        if username:
            smtp.login(username, password)

        smtp.send_message(message)
