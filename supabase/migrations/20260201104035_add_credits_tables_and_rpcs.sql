create extension if not exists pgcrypto;

create table if not exists public.credits_balance (
  user_id uuid primary key references auth.users(id) on delete cascade,
  balance integer not null default 0,
  updated_at timestamptz not null default now()
);

create table if not exists public.credits_ledger (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  change integer not null,
  reason text not null,
  job_id uuid null,
  meta jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now()
);

create index if not exists credits_ledger_user_id_created_at_idx
  on public.credits_ledger (user_id, created_at desc);

do $$
begin
  if to_regclass('public.audit_jobs') is not null then
    if not exists (
      select 1
      from information_schema.table_constraints
      where constraint_schema = 'public'
        and table_name = 'credits_ledger'
        and constraint_name = 'credits_ledger_job_id_fkey'
    ) then
      alter table public.credits_ledger
        add constraint credits_ledger_job_id_fkey
        foreign key (job_id) references public.audit_jobs(id) on delete set null;
    end if;
  end if;
end $$;

alter table public.credits_balance enable row level security;
alter table public.credits_ledger enable row level security;

do $$
begin
  if not exists (
    select 1 from pg_policies
    where schemaname = 'public'
      and tablename = 'credits_balance'
      and policyname = 'credits_balance_select_own'
  ) then
    create policy credits_balance_select_own
      on public.credits_balance
      for select
      to authenticated
      using (auth.uid() = user_id);
  end if;

  if not exists (
    select 1 from pg_policies
    where schemaname = 'public'
      and tablename = 'credits_ledger'
      and policyname = 'credits_ledger_select_own'
  ) then
    create policy credits_ledger_select_own
      on public.credits_ledger
      for select
      to authenticated
      using (auth.uid() = user_id);
  end if;
end $$;

create or replace function public.apply_credits(
  p_user_id uuid,
  p_change integer,
  p_reason text,
  p_job_id uuid default null,
  p_meta jsonb default '{}'::jsonb
)
returns table (ok boolean, balance integer)
language plpgsql
as $$
declare
  v_balance integer;
begin
  insert into public.credits_balance (user_id)
  values (p_user_id)
  on conflict (user_id) do nothing;

  select balance
  into v_balance
  from public.credits_balance
  where user_id = p_user_id
  for update;

  if p_change < 0 and (v_balance + p_change) < 0 then
    raise exception 'insufficient_credits';
  end if;

  update public.credits_balance
  set balance = v_balance + p_change,
      updated_at = now()
  where user_id = p_user_id
  returning balance into v_balance;

  insert into public.credits_ledger (
    user_id,
    change,
    reason,
    job_id,
    meta
  ) values (
    p_user_id,
    p_change,
    p_reason,
    p_job_id,
    p_meta
  );

  return query select true, v_balance;
end;
$$;

create or replace function public.get_credits_balance(
  p_user_id uuid
)
returns table (balance integer)
language sql
stable
as $$
  select coalesce(
    (select balance from public.credits_balance where user_id = p_user_id),
    0
  ) as balance;
$$;
