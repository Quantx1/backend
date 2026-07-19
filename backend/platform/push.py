"""
================================================================================
                    PUSH & EMAIL NOTIFICATION SERVICES
                    ===================================

    Web Push: VAPID-based browser push notifications (pywebpush)
    Email: Transactional emails via Resend API
================================================================================
"""

import json
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Web Push Service (VAPID)
# ---------------------------------------------------------------------------

class WebPushService:
    """Send Web Push notifications via VAPID protocol."""

    def __init__(self, vapid_private_key: str, vapid_claims_email: str):
        self.vapid_private_key = vapid_private_key
        self.vapid_claims_email = vapid_claims_email
        self._available = bool(vapid_private_key)
        if not self._available:
            logger.warning("WebPushService: VAPID_PRIVATE_KEY not set, push disabled")

    @property
    def is_available(self) -> bool:
        return self._available

    async def send(
        self,
        subscription_info: Dict,
        title: str,
        body: str,
        data: Optional[Dict] = None,
        tag: str = "default",
    ) -> bool:
        """
        Send a push notification to a single subscription.

        subscription_info: {
            "endpoint": "https://fcm.googleapis.com/...",
            "keys": {"p256dh": "...", "auth": "..."}
        }

        Returns True if sent successfully, False otherwise.
        Raises WebPushException with 410 status if subscription expired.
        """
        if not self._available:
            return False

        try:
            from pywebpush import webpush

            payload = json.dumps({
                "title": title,
                "body": body,
                "data": data or {},
                "tag": tag,
            })

            webpush(
                subscription_info=subscription_info,
                data=payload,
                vapid_private_key=self.vapid_private_key,
                vapid_claims={"sub": self.vapid_claims_email},
                ttl=86400,  # 24 hours
            )
            return True

        except Exception as e:
            # Check for 410 Gone (subscription expired)
            if hasattr(e, "response") and getattr(e.response, "status_code", 0) == 410:
                logger.info(f"Push subscription expired: {subscription_info.get('endpoint', '')[:60]}")
                raise  # Let caller handle cleanup
            logger.error(f"Web push error: {e}")
            return False


# ---------------------------------------------------------------------------
# Email Service (Resend)
# ---------------------------------------------------------------------------

class EmailService:
    """Send transactional emails via Resend API."""

    def __init__(self, api_key: str, from_email: str):
        self.api_key = api_key
        self.from_email = from_email
        self._available = bool(api_key)
        if not self._available:
            logger.warning("EmailService: RESEND_API_KEY not set, email disabled")

    @property
    def is_available(self) -> bool:
        return self._available

    async def send(self, to: str, subject: str, html: str) -> bool:
        """Send a single email. Returns True on success."""
        if not self._available:
            return False

        try:
            import resend
            resend.api_key = self.api_key

            resend.Emails.send({
                "from": self.from_email,
                "to": [to],
                "subject": subject,
                "html": html,
            })
            logger.info(f"Email sent to {to}: {subject}")
            return True

        except Exception as e:
            logger.error(f"Email send error to {to}: {e}")
            return False

    async def send_signal_alert(self, to: str, signal: Dict) -> bool:
        """Send a trading signal alert email."""
        symbol = signal.get("symbol", "N/A")
        direction = signal.get("direction", "LONG")
        entry = signal.get("entry_price", 0)
        target = signal.get("target", signal.get("target_1", 0))
        stop_loss = signal.get("stop_loss", 0)
        confidence = signal.get("confidence", 0)
        pattern = signal.get("pattern_type", signal.get("strategy", ""))

        direction_color = "#10B981" if direction == "LONG" else "#EF4444"
        direction_label = "BUY" if direction == "LONG" else "SELL"

        html = f"""
        <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; max-width: 480px; margin: 0 auto; background: #0F1117; color: #fff; border-radius: 12px; overflow: hidden;">
            <div style="background: linear-gradient(135deg, #4FECCD 0%, #3BD4AE 100%); padding: 20px 24px;">
                <h2 style="margin: 0; color: #0F1117; font-size: 18px;">New Signal: {symbol}</h2>
            </div>
            <div style="padding: 24px;">
                <div style="display: inline-block; background: {direction_color}20; color: {direction_color}; padding: 4px 12px; border-radius: 20px; font-size: 12px; font-weight: 600; margin-bottom: 16px;">{direction_label}</div>
                {f'<p style="color: #9CA3AF; font-size: 13px; margin: 0 0 16px;">Pattern: {pattern}</p>' if pattern else ''}
                <table style="width: 100%; border-collapse: collapse;">
                    <tr>
                        <td style="padding: 8px 0; color: #9CA3AF; font-size: 13px;">Entry</td>
                        <td style="padding: 8px 0; text-align: right; font-weight: 600;">&#8377;{entry:,.2f}</td>
                    </tr>
                    <tr>
                        <td style="padding: 8px 0; color: #10B981; font-size: 13px;">Target</td>
                        <td style="padding: 8px 0; text-align: right; font-weight: 600; color: #10B981;">&#8377;{target:,.2f}</td>
                    </tr>
                    <tr>
                        <td style="padding: 8px 0; color: #EF4444; font-size: 13px;">Stop Loss</td>
                        <td style="padding: 8px 0; text-align: right; font-weight: 600; color: #EF4444;">&#8377;{stop_loss:,.2f}</td>
                    </tr>
                    {f'<tr><td style="padding: 8px 0; color: #9CA3AF; font-size: 13px;">Confidence</td><td style="padding: 8px 0; text-align: right; font-weight: 600;">{confidence}%</td></tr>' if confidence else ''}
                </table>
                <div style="margin-top: 20px; text-align: center;">
                    <a href="https://app.quantx.app/signals" style="display: inline-block; background: #4FECCD; color: #0F1117; padding: 10px 24px; border-radius: 8px; text-decoration: none; font-weight: 600; font-size: 14px;">View Signal</a>
                </div>
            </div>
            <div style="padding: 12px 24px; border-top: 1px solid #1F2937; text-align: center;">
                <p style="color: #6B7280; font-size: 11px; margin: 0;">Quant X - AI-Powered Trading Signals</p>
            </div>
        </div>
        """

        return await self.send(to, f"Signal: {direction_label} {symbol}", html)

    async def send_position_alert(self, to: str, position: Dict, alert_type: str) -> bool:
        """Send SL hit or target hit email."""
        symbol = position.get("symbol", "N/A")
        entry = position.get("entry_price", 0)
        exit_price = position.get("exit_price", position.get("current_price", 0))
        pnl = position.get("pnl", 0)
        pnl_pct = position.get("pnl_percent", 0)

        is_target = alert_type == "target_hit"
        emoji = "🎯" if is_target else "⚠️"
        title = "Target Hit" if is_target else "Stop Loss Hit"
        title_color = "#10B981" if is_target else "#EF4444"
        pnl_color = "#10B981" if pnl >= 0 else "#EF4444"

        html = f"""
        <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; max-width: 480px; margin: 0 auto; background: #0F1117; color: #fff; border-radius: 12px; overflow: hidden;">
            <div style="background: {title_color}; padding: 20px 24px;">
                <h2 style="margin: 0; color: #fff; font-size: 18px;">{emoji} {title}: {symbol}</h2>
            </div>
            <div style="padding: 24px;">
                <table style="width: 100%; border-collapse: collapse;">
                    <tr>
                        <td style="padding: 8px 0; color: #9CA3AF; font-size: 13px;">Entry</td>
                        <td style="padding: 8px 0; text-align: right; font-weight: 600;">&#8377;{entry:,.2f}</td>
                    </tr>
                    <tr>
                        <td style="padding: 8px 0; color: #9CA3AF; font-size: 13px;">Exit</td>
                        <td style="padding: 8px 0; text-align: right; font-weight: 600;">&#8377;{exit_price:,.2f}</td>
                    </tr>
                    <tr style="border-top: 1px solid #1F2937;">
                        <td style="padding: 12px 0; color: #9CA3AF; font-size: 13px; font-weight: 600;">P&L</td>
                        <td style="padding: 12px 0; text-align: right; font-weight: 700; font-size: 16px; color: {pnl_color};">
                            {'+' if pnl >= 0 else ''}&#8377;{pnl:,.0f} ({'+' if pnl_pct >= 0 else ''}{pnl_pct:.1f}%)
                        </td>
                    </tr>
                </table>
                <div style="margin-top: 20px; text-align: center;">
                    <a href="https://app.quantx.app/trades" style="display: inline-block; background: #4FECCD; color: #0F1117; padding: 10px 24px; border-radius: 8px; text-decoration: none; font-weight: 600; font-size: 14px;">View Trades</a>
                </div>
            </div>
            <div style="padding: 12px 24px; border-top: 1px solid #1F2937; text-align: center;">
                <p style="color: #6B7280; font-size: 11px; margin: 0;">Quant X - AI-Powered Trading Signals</p>
            </div>
        </div>
        """

        return await self.send(to, f"{emoji} {title}: {symbol} | P&L: ₹{pnl:,.0f}", html)

    async def send_daily_summary(self, to: str, summary: Dict) -> bool:
        """Send end-of-day summary email."""
        total_pnl = summary.get("total_pnl", 0)
        win_rate = summary.get("win_rate", 0)
        trades_closed = summary.get("trades_closed", 0)
        open_positions = summary.get("open_positions", 0)

        pnl_color = "#10B981" if total_pnl >= 0 else "#EF4444"

        html = f"""
        <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; max-width: 480px; margin: 0 auto; background: #0F1117; color: #fff; border-radius: 12px; overflow: hidden;">
            <div style="background: linear-gradient(135deg, #4FECCD 0%, #3BD4AE 100%); padding: 20px 24px;">
                <h2 style="margin: 0; color: #0F1117; font-size: 18px;">Daily Trading Summary</h2>
            </div>
            <div style="padding: 24px;">
                <div style="text-align: center; margin-bottom: 20px;">
                    <p style="color: #9CA3AF; font-size: 12px; margin: 0 0 4px;">Today's P&L</p>
                    <p style="font-size: 28px; font-weight: 700; margin: 0; color: {pnl_color};">
                        {'+' if total_pnl >= 0 else ''}&#8377;{total_pnl:,.0f}
                    </p>
                </div>
                <table style="width: 100%; border-collapse: collapse;">
                    <tr>
                        <td style="padding: 10px 0; color: #9CA3AF; font-size: 13px;">Win Rate</td>
                        <td style="padding: 10px 0; text-align: right; font-weight: 600;">{win_rate:.1f}%</td>
                    </tr>
                    <tr>
                        <td style="padding: 10px 0; color: #9CA3AF; font-size: 13px;">Trades Closed</td>
                        <td style="padding: 10px 0; text-align: right; font-weight: 600;">{trades_closed}</td>
                    </tr>
                    <tr>
                        <td style="padding: 10px 0; color: #9CA3AF; font-size: 13px;">Open Positions</td>
                        <td style="padding: 10px 0; text-align: right; font-weight: 600;">{open_positions}</td>
                    </tr>
                </table>
                <div style="margin-top: 20px; text-align: center;">
                    <a href="https://app.quantx.app/dashboard" style="display: inline-block; background: #4FECCD; color: #0F1117; padding: 10px 24px; border-radius: 8px; text-decoration: none; font-weight: 600; font-size: 14px;">View Dashboard</a>
                </div>
            </div>
            <div style="padding: 12px 24px; border-top: 1px solid #1F2937; text-align: center;">
                <p style="color: #6B7280; font-size: 11px; margin: 0;">Quant X - AI-Powered Trading Signals</p>
            </div>
        </div>
        """

        return await self.send(to, f"Daily Summary | P&L: ₹{total_pnl:,.0f}", html)
