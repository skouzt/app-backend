-- Lily — personalization preferences
-- Run in the Supabase SQL editor. Safe to re-run (idempotent).
--
-- These were device-only (AsyncStorage) when the Personalization screen was built,
-- which meant they vanished on reinstall and never reached the model. Moving them
-- here makes them follow the account and lets prompts/lily_chat.py shape Lily's
-- voice from them.
--
-- Own table rather than columns on user_info: user_info is the onboarding intake
-- (written once, at signup), while these change whenever the user feels like it.
-- Keeping them apart means a preference edit can never disturb intake data.

create table if not exists public.user_personalization (
  user_id     text primary key,

  -- Base voice. Constrained because the value is interpolated into the system
  -- prompt; an unexpected string there is prompt injection with extra steps.
  tone        text        not null default 'Gentle'
              check (tone in ('Gentle', 'Friendly', 'Reflective', 'Direct')),

  -- {"warm":"More","brevity":"Less",...}. Values validated in the API layer;
  -- jsonb here so adding a trait does not need a migration.
  traits      jsonb       not null default '{}'::jsonb,

  -- Lily may open a conversation when it has been a while.
  check_ins   boolean     not null default true,

  -- Free text the user wrote about themselves. Length-capped so it cannot crowd
  -- out the conversation inside the model's context window.
  note        text        not null default '',
              constraint user_personalization_note_len check (char_length(note) <= 2000),

  created_at  timestamptz not null default now(),
  updated_at  timestamptz not null default now()
);

-- Every read is "this user's row", same as the rest of the schema.
create index if not exists user_personalization_user_id_idx
  on public.user_personalization (user_id);

-- The API uses the service-role key and scopes every query by user_id itself, but
-- RLS stays on so a future anon-key client cannot read across accounts.
alter table public.user_personalization enable row level security;

do $$
begin
  if not exists (
    select 1 from pg_policies
    where schemaname = 'public'
      and tablename  = 'user_personalization'
      and policyname = 'user_personalization_own_rows'
  ) then
    create policy user_personalization_own_rows
      on public.user_personalization
      for all
      using (user_id = auth.jwt() ->> 'sub')
      with check (user_id = auth.jwt() ->> 'sub');
  end if;
end $$;
