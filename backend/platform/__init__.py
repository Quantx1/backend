"""Platform layer — cross-cutting infra: notifications, events, scheduler,
system flags, admin audit, referrals.

Public sub-modules::

    from backend.platform.admin_audit   import audit_log
    from backend.platform.alerts        import channels_for_event
    from backend.platform.events        import emit_event, MessageType
    from backend.platform.push          import WebPushService, EmailService
    from backend.platform.realtime      import ConnectionManager, WSMessage
    from backend.platform.referrals     import attribute_referral, ...
    from backend.platform.scheduler     import SchedulerService
    from backend.platform.system_flags  import is_globally_halted, global_halt_reason
    from backend.platform.whatsapp      import WhatsAppService
"""
