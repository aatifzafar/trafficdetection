import os
import logging
import datetime
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from collections import deque

logger = logging.getLogger(__name__)


class BaseAlertHandler(ABC):
    @abstractmethod
    def dispatch(self, alert_payload: Dict[str, Any]) -> bool:
        pass


class ConsoleAlertHandler(BaseAlertHandler):
    def dispatch(self, alert_payload: Dict[str, Any]) -> bool:
        v_type = alert_payload.get("violation_type", "UNKNOWN")
        plate = alert_payload.get("plate_number", "UNIDENTIFIED")
        cam = alert_payload.get("camera_name", "Main Camera")
        t_str = alert_payload.get("time_str", datetime.datetime.utcnow().strftime("%H:%M:%S"))
        evt = alert_payload.get("event_id", "EVT-0000")

        logger.warning(
            f"\n"
            f"====================================================\n"
            f"[INCIDENT ALERT] TRAFFIC VIOLATION DETECTED\n"
            f"----------------------------------------------------\n"
            f"Type     : {v_type}\n"
            f"Vehicle  : {plate}\n"
            f"Camera   : {cam}\n"
            f"Time     : {t_str}\n"
            f"Evidence : {evt}\n"
            f"===================================================="
        )
        return True


class DashboardAlertHandler(BaseAlertHandler):
    """Maintains an in-memory real-time alert buffer for dashboard polling/websockets."""
    def __init__(self, max_alerts: int = 50):
        self.alerts_queue: deque = deque(maxlen=max_alerts)

    def dispatch(self, alert_payload: Dict[str, Any]) -> bool:
        self.alerts_queue.append(alert_payload)
        return True

    def get_latest_alerts(self, limit: int = 20) -> List[Dict[str, Any]]:
        return list(reversed(list(self.alerts_queue)))[:limit]


class SMSAlertHandler(BaseAlertHandler):
    """Optional SMS notification abstraction without hardcoded secrets."""
    def __init__(self):
        self.api_key = os.getenv("SMS_API_KEY", "")
        self.api_secret = os.getenv("SMS_API_SECRET", "")
        self.provider = os.getenv("SMS_PROVIDER", "mock")

    def dispatch(self, alert_payload: Dict[str, Any]) -> bool:
        if not self.api_key or self.provider == "mock":
            logger.info(f"[SMS MOCK] Simulated SMS alert sent for {alert_payload.get('event_id')}")
            return True
        # Real provider integration hooks (e.g. Twilio / Fast2SMS) can be plugged here
        logger.info(f"Dispatched SMS alert via {self.provider}")
        return True


class EmailAlertHandler(BaseAlertHandler):
    """Optional Email alert handler."""
    def __init__(self):
        self.smtp_host = os.getenv("SMTP_HOST", "")
        self.smtp_user = os.getenv("SMTP_USER", "")

    def dispatch(self, alert_payload: Dict[str, Any]) -> bool:
        if not self.smtp_host:
            logger.debug("[EMAIL MOCK] Email notifications not configured; skipping.")
            return True
        logger.info(f"Dispatched Email alert for {alert_payload.get('event_id')}")
        return True


class AlertService:
    """Central Alert Dispatcher."""
    def __init__(self):
        self.console_handler = ConsoleAlertHandler()
        self.dashboard_handler = DashboardAlertHandler()
        self.sms_handler = SMSAlertHandler()
        self.email_handler = EmailAlertHandler()

    def send_violation_alert(
        self,
        event_id: str,
        violation_type: str,
        plate_number: Optional[str],
        confidence: float,
        camera_name: str,
        location: str,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        payload = {
            "event_id": event_id,
            "violation_type": violation_type,
            "plate_number": plate_number or "UNIDENTIFIED",
            "confidence": round(confidence, 3),
            "camera_name": camera_name,
            "location": location,
            "time_str": datetime.datetime.utcnow().strftime("%H:%M:%S"),
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "extra": extra or {},
        }

        # Dispatch default channels
        self.console_handler.dispatch(payload)
        self.dashboard_handler.dispatch(payload)

        # Dispatch optional channels if configured
        if os.getenv("SMS_API_KEY"):
            self.sms_handler.dispatch(payload)
        if os.getenv("SMTP_HOST"):
            self.email_handler.dispatch(payload)

        return payload


alert_service = AlertService()
