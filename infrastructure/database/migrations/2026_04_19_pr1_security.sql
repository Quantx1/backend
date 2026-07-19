-- ============================================================================
-- PR 1 — P0 security fixes
-- ============================================================================
-- Adds is_admin column to user_profiles for role-based admin route gating.
-- Previously admin access was decided solely on ADMIN_EMAILS env var + an
-- unverified JWT email claim. With JWT signature verification now enabled
-- and this column in place, admin gating is cryptographically enforced.
--
-- Idempotent — safe to run multiple times.
-- ============================================================================

-- Add is_admin column (defaults to false — explicit promotion required)
alter table public.user_profiles
  add column if not exists is_admin boolean not null default false;

-- Index for the admin-gate hot path (WHERE is_admin = true is cheap + tiny).
create index if not exists user_profiles_is_admin_idx
  on public.user_profiles (is_admin)
  where is_admin = true;

-- Bootstrap: promote anyone already in ADMIN_EMAILS env var list.
-- The ADMIN_EMAILS list is read at runtime by get_admin_user() as a fallback,
-- so first-time deploy still works even if this migration hasn't flipped the
-- column yet. This UPDATE is just a convenience to reflect current admins.
--
-- To use: set app.admin_email_bootstrap in Postgres to your admin email(s),
-- e.g.
--   select set_config('app.admin_email_bootstrap', 'you@example.com', false);
-- then run the UPDATE below. Commented by default to keep this migration
-- side-effect-free.
--
-- update public.user_profiles
-- set is_admin = true
-- where email = current_setting('app.admin_email_bootstrap', true)
--   and is_admin = false;

-- Log migration
insert into public.schema_migrations (version, description)
values ('2026_04_19_pr1_security', 'PR 1 P0 security — adds is_admin column to user_profiles')
on conflict (version) do nothing;
