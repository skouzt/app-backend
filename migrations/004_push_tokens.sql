-- ─────────────────────────────────────────────────────────────────────────────
-- Push notification tokens.
--
-- One row per device, not per user: a person with a phone and a tablet gets two,
-- and both should hear from Lily.
-- ─────────────────────────────────────────────────────────────────────────────

create table if not exists public.push_tokens (
  -- The Expo push token is the natural key. It identifies an app install on a
  -- device, and it is what the send call addresses.
  --
  -- Primary key rather than a surrogate id because the same token must never
  -- exist twice: registering is an upsert, and a device that is handed to
  -- another account moves to that user rather than accumulating a second row
  -- that would push someone else's notification to the same phone.
  token       text primary key,

  -- Clerk's user id, matching every other table here. Not a foreign key because
  -- users live in Clerk, not in this database.
  user_id     text not null,

  platform    text not null check (platform in ('ios', 'android')),

  -- The Settings toggle. Rows are kept when a user opts out rather than deleted,
  -- so turning notifications back on does not depend on the device handing us a
  -- fresh token first — Expo only reissues one when the install changes.
  enabled     boolean not null default true,

  created_at  timestamptz not null default now(),
  updated_at  timestamptz not null default now()
);

-- Sending starts from "which devices belong to this user", so that is the index.
create index if not exists push_tokens_user_id_idx
  on public.push_tokens (user_id)
  where enabled;

-- ─────────────────────────────────────────────────────────────────────────────
-- The API uses the service-role key and scopes every query by user_id itself,
-- but RLS stays on so a future anon-key client cannot read across accounts.
-- A push token is enough to send someone a notification, so this table should
-- never be readable by anyone but its owner.
-- ─────────────────────────────────────────────────────────────────────────────
alter table public.push_tokens enable row level security;

do $$
begin
  if not exists (
    select 1 from pg_policies
    where schemaname = 'public'
      and tablename  = 'push_tokens'
      and policyname = 'push_tokens_own_rows'
  ) then
    create policy push_tokens_own_rows
      on public.push_tokens
      for all
      using (user_id = auth.jwt() ->> 'sub')
      with check (user_id = auth.jwt() ->> 'sub');
  end if;
end $$;
