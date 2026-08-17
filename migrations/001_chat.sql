-- Lily — chat schema + webhook idempotency
-- Run in the Supabase SQL editor. Safe to re-run (idempotent).
--
-- Design note: chat sessions extend `therapy_sessions` rather than living in a new
-- table. That keeps the existing /therapy/sessions and /therapy/journey endpoints
-- (and every user's history) working unchanged, and avoids dual-writing summaries.

-- ─────────────────────────────────────────────────────────────────────────────
-- 1. webhook_events — MISSING IN PRODUCTION.
--    api/v1/billing/dodo.py queries this before dispatching any Dodo event, so
--    every webhook has been 500ing. Creating it restores renewal / cancellation /
--    expiry handling.
-- ─────────────────────────────────────────────────────────────────────────────

create table if not exists public.webhook_events (
  id          text primary key,
  received_at timestamptz not null default now()
);

alter table public.webhook_events enable row level security;

-- ─────────────────────────────────────────────────────────────────────────────
-- 2. therapy_sessions — becomes the chat session record.
--    Existing rows are historical voice sessions; they default to 'ended' so the
--    reaper never picks them up.
-- ─────────────────────────────────────────────────────────────────────────────

alter table public.therapy_sessions
  add column if not exists status          text        not null default 'ended',
  add column if not exists last_message_at timestamptz,
  add column if not exists message_count   integer     not null default 0;

do $$
begin
  if not exists (
    select 1 from pg_constraint where conname = 'therapy_sessions_status_check'
  ) then
    alter table public.therapy_sessions
      add constraint therapy_sessions_status_check
      check (status in ('active', 'closing', 'ended'));
  end if;
end $$;

-- A user may only ever have one *active* conversation. Correctness guard, not an
-- optimisation: without it two concurrent sends can each create a session.
--
-- Deliberately scoped to 'active' only. A session that has gone idle is flipped to
-- 'closing' and left for the reaper to summarise; if this index also covered
-- 'closing', that pending row would block the user from starting their next
-- conversation until the summary finished.
create unique index if not exists therapy_sessions_one_active_per_user
  on public.therapy_sessions (user_id)
  where status = 'active';

-- Reaper scan: find sessions idle past the cutoff.
create index if not exists therapy_sessions_reaper
  on public.therapy_sessions (status, last_message_at)
  where status in ('active', 'closing');

-- History / journey listing.
create index if not exists therapy_sessions_user_created
  on public.therapy_sessions (user_id, created_at desc);

-- ─────────────────────────────────────────────────────────────────────────────
-- 3. messages — the turn-by-turn transcript.
-- ─────────────────────────────────────────────────────────────────────────────

create table if not exists public.messages (
  id         uuid primary key default gen_random_uuid(),
  session_id uuid not null references public.therapy_sessions(id) on delete cascade,
  user_id    text not null,
  role       text not null check (role in ('user', 'lily')),
  content    text not null,
  created_at timestamptz not null default now()
);

-- Chat scrollback is paginated by time across sessions, not within one.
create index if not exists messages_user_created
  on public.messages (user_id, created_at desc);

-- Loading one session's thread for summarisation.
create index if not exists messages_session_created
  on public.messages (session_id, created_at);

alter table public.messages enable row level security;

-- ─────────────────────────────────────────────────────────────────────────────
-- 4. RLS
--    The API connects with the service-role key, which bypasses RLS entirely.
--    Enabling it with no policies means the anon key can read nothing — these
--    tables hold private conversations, so default-deny is the right posture.
-- ─────────────────────────────────────────────────────────────────────────────

alter table public.therapy_sessions enable row level security;

-- ─────────────────────────────────────────────────────────────────────────────
-- 5. daily_usage is retired.
--    Billing is now a subscription gate with no metering. Left in place so no
--    data is destroyed — drop it once you're happy nothing reads it:
--
--    drop table if exists public.daily_usage;
-- ─────────────────────────────────────────────────────────────────────────────
