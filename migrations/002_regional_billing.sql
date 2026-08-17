-- Lily — regional billing
-- Run in the Supabase SQL editor. Idempotent.
--
-- plan_key moves from clarity|insight (metered by sessions) to monthly|yearly
-- (unlimited). Region and currency are recorded so we can tell what a subscriber
-- was actually charged, and detect when the tier sold doesn't match where they paid
-- from.

alter table public.dodo_subscriptions
  add column if not exists region      text,
  add column if not exists currency    text,
  add column if not exists amount      numeric,
  add column if not exists region_source text;

alter table public.pending_verifications
  add column if not exists region text;

-- Existing rows predate regional pricing; they were all USD-tier.
update public.dodo_subscriptions
   set region = 'INTL', currency = 'USD'
 where region is null;

comment on column public.dodo_subscriptions.region_source is
  'How the billing region was determined: an edge geo header (trusted) or client-hint (spoofable).';
