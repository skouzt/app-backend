-- ─────────────────────────────────────────────────────────────────────────────
-- 1. user_info gets one row per user.
--
--    onboarding-submit inserted rather than upserted, and nothing stopped a
--    second row for the same user_id. Anyone who ran onboarding twice — after a
--    failed status check, a reinstall, or a double tap on Finish — left a second
--    row behind. There are five such users today, one of them with two different
--    names seven months apart.
--
--    That matters because fetch_user_info() selects with limit(1) and no order:
--    Postgres may hand back either row, so Lily could greet someone by a name
--    they replaced. Deduplicating and constraining the column removes the
--    ambiguity rather than papering over it at read time.
-- ─────────────────────────────────────────────────────────────────────────────

-- Keep the newest row per user: it holds the answers they most recently gave.
-- id breaks ties for the pair written 440ms apart by a double submit, where
-- created_at alone may not separate them.
delete from public.user_info a
using public.user_info b
where a.user_id = b.user_id
  and (a.created_at < b.created_at
       or (a.created_at = b.created_at and a.id < b.id));

do $$
begin
  alter table public.user_info
    add constraint user_info_user_id_key unique (user_id);
exception
  when duplicate_object then null;
end $$;

-- The unique constraint creates its own index on user_id, so the old plain one
-- is now redundant — two identical btrees maintained on every write.
drop index if exists public.idx_user_info_user_id;
