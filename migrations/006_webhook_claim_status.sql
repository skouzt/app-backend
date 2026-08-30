-- ─────────────────────────────────────────────────────────────────────────────
-- 1. webhook_events gains a claim status.
--
--    The table recorded only that an event id had been seen, which cannot tell
--    "handled successfully" apart from "started and never finished". A process
--    killed mid-dispatch — OOM, a container replaced during deploy — left a row
--    behind that made every later redelivery look like a duplicate, so the event
--    was dropped. For subscription.active that is a paid subscription the
--    database never hears about.
--
--    With a status, an abandoned claim is recognisable and can be taken over by
--    a later delivery.
-- ─────────────────────────────────────────────────────────────────────────────

alter table public.webhook_events
  add column if not exists status     text,
  add column if not exists claimed_at timestamptz not null default now();

-- Existing rows were written by the old code, which only ever inserted after
-- deciding to handle an event. Treat them as finished: marking them
-- 'processing' would make the reaper hand every historical event back for
-- reprocessing once its claim aged out.
update public.webhook_events
set status = 'done'
where status is null;

alter table public.webhook_events
  alter column status set default 'processing',
  alter column status set not null;

-- Two states only. A typo in application code should fail the write rather than
-- create a third status that nothing knows how to interpret — an unrecognised
-- value would never be reclaimed and would silently swallow the event.
do $$
begin
  alter table public.webhook_events
    add constraint webhook_events_status_check
    check (status in ('processing', 'done'));
exception
  when duplicate_object then null;
end $$;

-- Supports the staleness lookup: only live claims are ever scanned, and the
-- partial predicate keeps the index to the handful of rows in flight rather
-- than every event ever received.
create index if not exists webhook_events_stale_claims
  on public.webhook_events (claimed_at)
  where status = 'processing';
