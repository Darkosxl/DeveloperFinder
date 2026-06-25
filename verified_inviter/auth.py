from __future__ import annotations

import functools

from authlib.integrations.flask_client import OAuth
from flask import (
    Blueprint,
    Flask,
    redirect,
    render_template,
    session,
    url_for,
)

from verified_inviter import config

oauth = OAuth()


def register_google_oauth(app: Flask) -> None:
    """Initialize Authlib OAuth and register the Google client."""
    oauth.init_app(app)
    oauth.register(
        name="google",
        client_id=config.GOOGLE_CLIENT_ID,
        client_secret=config.GOOGLE_CLIENT_SECRET,
        server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
        client_kwargs={"scope": "openid email profile"},
    )


auth_bp = Blueprint("auth", __name__, url_prefix="")


@auth_bp.route("/login")
def login():
    return render_template("login.html")


@auth_bp.route("/login/google")
def login_google():
    redirect_uri = url_for("auth.callback", _external=True)
    return oauth.google.authorize_redirect(redirect_uri)


@auth_bp.route("/auth/google/callback")
def callback():
    token = oauth.google.authorize_access_token()
    userinfo = token.get("userinfo")
    if not userinfo:
        resp = oauth.google.get("https://openidconnect.googleapis.com/v1/userinfo")
        userinfo = resp.json()
    email = userinfo.get("email")
    if email != config.ALLOWED_EMAIL:
        return render_template("access_denied.html", email=email), 403
    session["email"] = email
    session["authenticated"] = True
    return redirect(url_for("overview"))


@auth_bp.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("auth.login"))


def require_auth(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if not session.get("authenticated"):
            return redirect(url_for("auth.login"))
        return fn(*args, **kwargs)

    return wrapper


def configure_auth(app: Flask) -> None:
    """Register the auth blueprint and initialize Google OAuth on the app."""
    register_google_oauth(app)
    app.register_blueprint(auth_bp)
