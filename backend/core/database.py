"""
Database client and connection management
"""
from supabase import create_client, Client
from functools import lru_cache
from .config import settings

# In tests / any env-less context (CI without secrets) SUPABASE_URL can be empty,
# and supabase-py raises "supabase_url is required" — which would crash at import
# since the singletons below are built eagerly. Fall back to a syntactically valid
# placeholder so importing this module never throws; with real env injected at
# runtime the real values are used (and startup secret validation in config.py
# still fails loudly if required secrets are missing in production).
_PLACEHOLDER_URL = "https://placeholder.supabase.co"
_PLACEHOLDER_KEY = "placeholder-key"


@lru_cache()
def get_supabase_client() -> Client:
    """Get Supabase client (anon key)"""
    return create_client(
        settings.SUPABASE_URL or _PLACEHOLDER_URL,
        settings.SUPABASE_ANON_KEY or _PLACEHOLDER_KEY,
    )


@lru_cache()
def get_supabase_admin() -> Client:
    """Get Supabase admin client (service role key)"""
    return create_client(
        settings.SUPABASE_URL or _PLACEHOLDER_URL,
        settings.SUPABASE_SERVICE_KEY or _PLACEHOLDER_KEY,
    )


# Singleton instances
supabase = get_supabase_client()
supabase_admin = get_supabase_admin()
