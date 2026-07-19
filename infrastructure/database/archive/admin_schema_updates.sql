-- ============================================================================
-- QUANT X ADMIN SCHEMA UPDATES
-- Run this migration to add admin support
-- ============================================================================

-- ============================================================================
-- 1. ADD ADMIN FIELDS TO USER_PROFILES
-- ============================================================================

-- Add suspension fields
ALTER TABLE public.user_profiles 
ADD COLUMN IF NOT EXISTS is_suspended BOOLEAN DEFAULT FALSE;

ALTER TABLE public.user_profiles 
ADD COLUMN IF NOT EXISTS suspended_at TIMESTAMPTZ;

ALTER TABLE public.user_profiles 
ADD COLUMN IF NOT EXISTS suspended_by UUID;

ALTER TABLE public.user_profiles 
ADD COLUMN IF NOT EXISTS suspension_reason TEXT;

-- Add ban fields
ALTER TABLE public.user_profiles 
ADD COLUMN IF NOT EXISTS is_banned BOOLEAN DEFAULT FALSE;

ALTER TABLE public.user_profiles 
ADD COLUMN IF NOT EXISTS banned_at TIMESTAMPTZ;

ALTER TABLE public.user_profiles 
ADD COLUMN IF NOT EXISTS banned_by UUID;

ALTER TABLE public.user_profiles 
ADD COLUMN IF NOT EXISTS ban_reason TEXT;

-- ============================================================================
-- 2. CREATE ADMIN_USERS TABLE
-- ============================================================================

CREATE TABLE IF NOT EXISTS public.admin_users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES public.user_profiles(id) ON DELETE CASCADE,
    email TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('super_admin', 'support_admin', 'read_only')),
    permissions JSONB DEFAULT '{}',
    
    -- Status
    is_active BOOLEAN DEFAULT TRUE,
    
    -- Audit
    created_at TIMESTAMPTZ DEFAULT NOW(),
    created_by UUID,
    last_login TIMESTAMPTZ,
    
    UNIQUE(user_id)
);

CREATE INDEX IF NOT EXISTS idx_admin_users_email ON public.admin_users(email);
CREATE INDEX IF NOT EXISTS idx_admin_users_role ON public.admin_users(role);

-- ============================================================================
-- 3. ADD REFUND FIELDS TO PAYMENTS
-- ============================================================================

ALTER TABLE public.payments 
ADD COLUMN IF NOT EXISTS refund_id TEXT;

ALTER TABLE public.payments 
ADD COLUMN IF NOT EXISTS refund_amount INTEGER;

ALTER TABLE public.payments 
ADD COLUMN IF NOT EXISTS refund_reason TEXT;

ALTER TABLE public.payments 
ADD COLUMN IF NOT EXISTS refunded_at TIMESTAMPTZ;

-- ============================================================================
-- 4. ADMIN AUDIT LOG ENHANCEMENTS
-- ============================================================================

-- Add more fields to audit_log if needed
ALTER TABLE public.audit_log 
ADD COLUMN IF NOT EXISTS admin_id UUID;

ALTER TABLE public.audit_log 
ADD COLUMN IF NOT EXISTS admin_email TEXT;

-- Create index for admin actions
CREATE INDEX IF NOT EXISTS idx_audit_log_admin ON public.audit_log(admin_id, created_at DESC);

-- ============================================================================
-- 5. RLS POLICIES FOR ADMIN ACCESS
-- ============================================================================

-- Enable RLS on admin_users
ALTER TABLE public.admin_users ENABLE ROW LEVEL SECURITY;

-- Only admins can view admin_users table
CREATE POLICY "Admins can view admin users" ON public.admin_users
    FOR SELECT
    USING (
        EXISTS (
            SELECT 1 FROM public.admin_users au 
            WHERE au.user_id = auth.uid() AND au.is_active = TRUE
        )
    );

-- Only super_admin can modify admin_users
CREATE POLICY "Super admins can manage admin users" ON public.admin_users
    FOR ALL
    USING (
        EXISTS (
            SELECT 1 FROM public.admin_users au 
            WHERE au.user_id = auth.uid() 
            AND au.role = 'super_admin' 
            AND au.is_active = TRUE
        )
    );

-- Admin access to user_profiles (read all)
CREATE POLICY "Admins can view all profiles" ON public.user_profiles
    FOR SELECT
    USING (
        auth.uid() = id OR
        EXISTS (
            SELECT 1 FROM public.admin_users au 
            WHERE au.user_id = auth.uid() AND au.is_active = TRUE
        )
    );

-- Admin access to modify user_profiles (support_admin, super_admin only)
CREATE POLICY "Support admins can update profiles" ON public.user_profiles
    FOR UPDATE
    USING (
        auth.uid() = id OR
        EXISTS (
            SELECT 1 FROM public.admin_users au 
            WHERE au.user_id = auth.uid() 
            AND au.role IN ('super_admin', 'support_admin')
            AND au.is_active = TRUE
        )
    );

-- Admin access to payments (read)
CREATE POLICY "Admins can view all payments" ON public.payments
    FOR SELECT
    USING (
        auth.uid() = user_id OR
        EXISTS (
            SELECT 1 FROM public.admin_users au 
            WHERE au.user_id = auth.uid() AND au.is_active = TRUE
        )
    );

-- Admin access to trades (read)
CREATE POLICY "Admins can view all trades" ON public.trades
    FOR SELECT
    USING (
        auth.uid() = user_id OR
        EXISTS (
            SELECT 1 FROM public.admin_users au 
            WHERE au.user_id = auth.uid() AND au.is_active = TRUE
        )
    );

-- Admin access to positions (read)
CREATE POLICY "Admins can view all positions" ON public.positions
    FOR SELECT
    USING (
        auth.uid() = user_id OR
        EXISTS (
            SELECT 1 FROM public.admin_users au 
            WHERE au.user_id = auth.uid() AND au.is_active = TRUE
        )
    );

-- ============================================================================
-- 6. HELPER FUNCTION: CHECK ADMIN ROLE
-- ============================================================================

CREATE OR REPLACE FUNCTION public.is_admin(user_uuid UUID DEFAULT auth.uid())
RETURNS BOOLEAN AS $$
BEGIN
    RETURN EXISTS (
        SELECT 1 FROM public.admin_users 
        WHERE user_id = user_uuid AND is_active = TRUE
    );
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

CREATE OR REPLACE FUNCTION public.get_admin_role(user_uuid UUID DEFAULT auth.uid())
RETURNS TEXT AS $$
DECLARE
    admin_role TEXT;
BEGIN
    SELECT role INTO admin_role 
    FROM public.admin_users 
    WHERE user_id = user_uuid AND is_active = TRUE;
    
    RETURN admin_role;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- ============================================================================
-- 7. GRANTS
-- ============================================================================

GRANT ALL ON public.admin_users TO authenticated;
GRANT ALL ON public.admin_users TO service_role;
GRANT EXECUTE ON FUNCTION public.is_admin TO authenticated;
GRANT EXECUTE ON FUNCTION public.get_admin_role TO authenticated;
