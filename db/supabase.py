import os
from supabase import create_client, Client
from supabase.client import ClientOptions

from dotenv import load_dotenv
load_dotenv()

SUPABASE_SERVICE_ROLE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
SUPABASE_URL = os.environ["SUPABASE_URL"]
supabase: Client = create_client(
    SUPABASE_URL,
    SUPABASE_SERVICE_ROLE_KEY,
    options=ClientOptions(
        postgrest_client_timeout=10,
        storage_client_timeout=10,
        schema="public",
    )
)


