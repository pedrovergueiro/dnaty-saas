"""
Shared WebSocket manager for admin live feed.
Imported by routes/admin.py and any endpoint that emits events.
"""
import logging
from datetime import datetime
from typing import List

from fastapi import WebSocket

logger = logging.getLogger(__name__)


class AdminWSManager:
    def __init__(self) -> None:
        self.connections: List[WebSocket] = []

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self.connections.append(ws)
        logger.debug("Admin WS connected (%d total)", len(self.connections))

    def disconnect(self, ws: WebSocket) -> None:
        if ws in self.connections:
            self.connections.remove(ws)
        logger.debug("Admin WS disconnected (%d remaining)", len(self.connections))

    async def broadcast(self, event: dict) -> None:
        dead: List[WebSocket] = []
        for ws in self.connections:
            try:
                await ws.send_json(event)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)


ws_manager = AdminWSManager()


async def emit_event(event_type: str, data: dict) -> None:
    """Fire-and-forget: emit a typed event to all connected admin sessions."""
    await ws_manager.broadcast({
        "type": event_type,
        "data": data,
        "timestamp": datetime.utcnow().isoformat(),
    })
