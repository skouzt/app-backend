from pydantic_settings import BaseSettings
from pydantic import ConfigDict


class Settings(BaseSettings):
    # extra="allow" so an unrecognised .env key is ignored rather than fatal.
    model_config = ConfigDict(
        env_file=".env",
        extra="allow",
    )

    # Note: SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY are read directly from the
    # environment in db/supabase.py, not through here — they must be present at
    # import time, and a silent "" default would defer that failure to the first
    # query. There was also a dead SUPABASE_SERVICE_KEY field here, differing from
    # the real variable by one word and defaulting to empty.

    # --- Dodo Payments ---
    DODO_PAYMENTS_API_KEY: str = ""
    DODO_WEBHOOK_SECRET: str = ""
    DODO_ENVIRONMENT: str = "test_mode"
    DODO_DEFAULT_RETURN_URL: str = "aletheia://payment/result"

    # One Dodo product per billing interval. Each carries per-country prices via
    # Dodo's Localized Pricing, so currency is resolved by Dodo from the customer's
    # country — we don't need a product per currency.
    DODO_PRODUCT_MONTHLY: str = ""
    DODO_PRODUCT_YEARLY: str = ""

    # --- Chat LLM ---
    DEEPSEEK_API_KEY: str = ""
    # Pinned explicitly: "deepseek-chat" is a legacy alias that still resolves but
    # is no longer listed, so you cannot tell which model you are billed for.
    DEEPSEEK_MODEL: str = "deepseek-v4-flash"
    DEEPSEEK_BASE_URL: str = "https://api.deepseek.com/v1"

    # --- Security ---
    # Empty by default, which denies all cross-origin browser requests. The only
    # client is the native app, which is not subject to CORS at all — so a wildcard
    # bought nothing and let any website call this API from a user's browser with a
    # token it had scraped. Set to a comma-separated origin list only when a real
    # web client exists.
    ALLOWED_ORIGINS: str = ""


settings = Settings()
