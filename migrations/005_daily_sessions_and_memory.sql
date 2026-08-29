-- ─────────────────────────────────────────────────────────────────────────────
-- 1. Summary watermark on therapy_sessions.
--
--    A day's session is now summarised repeatedly as the day goes on rather than
--    once when it closes, so something has to distinguish "the conversation moved
--    on" from "nothing has happened since". Without it every reaper sweep would
--    re-run the model over an unchanged thread — paying for the same summary and
--    rewriting one the user may already have read.
-- ─────────────────────────────────────────────────────────────────────────────

alter table public.therapy_sessions
  add column if not exists summarised_message_count integer not null default 0;

-- Rows that already carry a summary predate the watermark and would otherwise
-- look stale, drawing one pointless re-summarisation each.
update public.therapy_sessions
set summarised_message_count = coalesce(message_count, 0)
where summary is not null
  and summarised_message_count = 0;

-- ─────────────────────────────────────────────────────────────────────────────
-- 2. user_memory — what Lily carries forward about someone.
--
--    Memory was the last few session summaries, so it had a horizon: a sister
--    mentioned in March was gone by April. This is the durable half — facts that
--    stay true regardless of when they were said.
--
--    One row per user, grouped by kind rather than one blob, so the prompt can
--    render sections and each kind can be capped on its own. Extraction rewrites
--    the whole set rather than appending, because people revise: someone who
--    left a job needs the old fact gone, not filed next to the new one.
-- ─────────────────────────────────────────────────────────────────────────────

create table if not exists public.user_memory (
  user_id     text primary key,
  facts       jsonb       not null default '{}'::jsonb,
  last_session_id uuid,
  created_at  timestamptz not null default now(),
  updated_at  timestamptz not null default now()
);

-- The most sensitive table here: a written record of someone's family, work and
-- state of mind. The API scopes by user_id with the service-role key, but RLS
-- stays on so no anon-key client can read across accounts.
alter table public.user_memory enable row level security;

do $$
begin
  if not exists (
    select 1 from pg_policies
    where schemaname = 'public'
      and tablename  = 'user_memory'
      and policyname = 'user_memory_own_rows'
  ) then
    create policy user_memory_own_rows
      on public.user_memory
      for all
      using (user_id = auth.jwt() ->> 'sub')
      with check (user_id = auth.jwt() ->> 'sub');
  end if;
end $$;
