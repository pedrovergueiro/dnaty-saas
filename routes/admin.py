"""
Admin dashboard endpoints — all protected by require_admin JWT dependency.

All endpoints return 404 (never 401/403) when accessed without proper authentication,
emitting admin events to WebSocket connections when operations occur.
"""
import logging
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Request, WebSocket
from fastapi.responses import JSONResponse

from config import settings
from models.admin_store import (
    append_audit_log,
    admin_ban_user,
    admin_delete_user,
    admin_unban_user,
    change_user_plan,
    create_license,
    get_overview_stats,
    get_revenue_chart,
    get_user_detail,
    list_audit_log,
    list_bans,
    list_failed_logins,
    list_licenses,
    list_trainings,
    list_users,
)
from models.ws_manager import emit_event, ws_manager
from routes.admin_auth import require_admin, validate_admin_token, _get_client_ip
from routes import train as train_routes
from models.dnaty_model import _jobs

logger = logging.getLogger(__name__)
router = APIRouter()


# ── Overview ───────────────────────────────────────────────────────────────────

@router.get("/admin/overview", summary="Admin dashboard overview")
async def overview(
    request: Request,
    _: str = Depends(require_admin),
):
    """Get KPI cards and live job count."""
    live_job_count = len(_jobs)
    stats = get_overview_stats(live_jobs=live_job_count)
    return stats


# ── Users ──────────────────────────────────────────────────────────────────────

@router.get("/admin/users", summary="List users with pagination and filters")
async def list_users_endpoint(
    request: Request,
    page: int = 1,
    limit: int = 50,
    plan: str = "all",
    search: str = "",
    _: str = Depends(require_admin),
):
    """List users with optional filtering by plan and search."""
    result = list_users(page=page, limit=limit, plan=plan, search=search)
    return result


@router.get("/admin/users/{email}", summary="Get detailed user information")
async def user_detail(
    email: str,
    request: Request,
    _: str = Depends(require_admin),
):
    """Retrieve full user details including keys, trainings, fingerprints, bans."""
    detail = get_user_detail(email)
    if not detail:
        raise HTTPException(status_code=404, detail="Not found")
    return detail


@router.patch("/admin/users/{email}/plan", summary="Change user subscription plan")
async def change_plan(
    email: str,
    request: Request,
    body: dict,  # {plan: "pro"|"free"}
    _: str = Depends(require_admin),
):
    """Change user plan and emit event."""
    ip = _get_client_ip(request)
    plan = body.get("plan", "free")
    
    if plan not in ("pro", "free"):
        raise HTTPException(status_code=400, detail="Invalid plan")
    
    ok = change_user_plan(email, plan)
    if not ok:
        raise HTTPException(status_code=404, detail="Not found")
    
    append_audit_log(
        action="change_user_plan",
        target_type="user",
        target_id=email,
        details={"plan": plan},
        ip=ip,
    )
    
    await emit_event("user_plan_changed", {
        "email": email,
        "plan": plan,
        "timestamp": datetime.utcnow().isoformat(),
    })
    
    return {"email": email, "plan": plan}


@router.post("/admin/users/{email}/ban", summary="Ban a user")
async def ban_user_endpoint(
    email: str,
    request: Request,
    body: dict,  # {reason}
    _: str = Depends(require_admin),
):
    """Ban a user by email and emit event."""
    ip = _get_client_ip(request)
    reason = body.get("reason", "admin ban")
    
    admin_ban_user(email, reason=reason)
    
    append_audit_log(
        action="ban_user",
        target_type="user",
        target_id=email,
        details={"reason": reason},
        ip=ip,
    )
    
    await emit_event("user_banned", {
        "email": email,
        "reason": reason,
        "timestamp": datetime.utcnow().isoformat(),
    })
    
    return {"email": email, "banned": True}


@router.post("/admin/users/{email}/unban", summary="Unban a user")
async def unban_user_endpoint(
    email: str,
    request: Request,
    _: str = Depends(require_admin),
):
    """Unban a user by email and emit event."""
    ip = _get_client_ip(request)
    
    admin_unban_user(email)
    
    append_audit_log(
        action="unban_user",
        target_type="user",
        target_id=email,
        ip=ip,
    )
    
    await emit_event("user_unbanned", {
        "email": email,
        "timestamp": datetime.utcnow().isoformat(),
    })
    
    return {"email": email, "unbanned": True}


@router.delete("/admin/users/{email}", summary="Delete a user")
async def delete_user_endpoint(
    email: str,
    request: Request,
    _: str = Depends(require_admin),
):
    """Delete a user and emit event."""
    ip = _get_client_ip(request)
    
    ok = admin_delete_user(email)
    if not ok:
        raise HTTPException(status_code=404, detail="Not found")
    
    append_audit_log(
        action="delete_user",
        target_type="user",
        target_id=email,
        ip=ip,
    )
    
    await emit_event("user_deleted", {
        "email": email,
        "timestamp": datetime.utcnow().isoformat(),
    })
    
    return {"email": email, "deleted": True}


# ── Trainings ──────────────────────────────────────────────────────────────────

@router.get("/admin/trainings", summary="List training history")
async def trainings_endpoint(
    request: Request,
    page: int = 1,
    limit: int = 50,
    user_email: str = "",
    _: str = Depends(require_admin),
):
    """List training usage logs with pagination."""
    result = list_trainings(page=page, limit=limit, user_email=user_email)
    return result


@router.get("/admin/trainings/live", summary="List currently running jobs")
async def live_jobs_endpoint(
    request: Request,
    _: str = Depends(require_admin),
):
    """Return in-memory job list."""
    jobs = []
    for job_id, job_data in _jobs.items():
        jobs.append({
            "id": job_id,
            "user_email": job_data.get("params", {}).get("user_email", "unknown"),
            "status": job_data.get("status"),
            "progress": job_data.get("progress"),
            "current_generation": job_data.get("current_generation"),
            "total_generations": job_data.get("total_generations"),
        })
    return {"jobs": jobs, "total": len(jobs)}


# ── Security ───────────────────────────────────────────────────────────────────

@router.get("/admin/security/bans", summary="List user bans")
async def bans_endpoint(
    request: Request,
    page: int = 1,
    limit: int = 50,
    active_only: bool = True,
    _: str = Depends(require_admin),
):
    """List bans with pagination."""
    result = list_bans(page=page, limit=limit, active_only=active_only)
    return result


@router.post("/admin/security/bans/{ban_id}/unban", summary="Unban by ban ID")
async def unban_by_ban_id(
    ban_id: str,
    request: Request,
    _: str = Depends(require_admin),
):
    """Unban using ban ID and emit event."""
    from models.abuse_store import unban_by_id
    
    ip = _get_client_ip(request)
    ok = unban_by_id(ban_id)
    
    if not ok:
        raise HTTPException(status_code=404, detail="Not found")
    
    append_audit_log(
        action="unban_by_id",
        target_type="ban",
        target_id=ban_id,
        ip=ip,
    )
    
    await emit_event("ban_revoked", {
        "ban_id": ban_id,
        "timestamp": datetime.utcnow().isoformat(),
    })
    
    return {"ban_id": ban_id, "unbanned": True}


@router.get("/admin/security/failed-logins", summary="Recent failed login attempts")
async def failed_logins_endpoint(
    request: Request,
    hours: int = 24,
    _: str = Depends(require_admin),
):
    """List failed admin login attempts."""
    logins = list_failed_logins(hours=hours)
    return {"failed_logins": logins, "total": len(logins)}


# ── Revenue ────────────────────────────────────────────────────────────────────

@router.get("/admin/revenue/chart", summary="MRR chart (30 days)")
async def revenue_chart_endpoint(
    request: Request,
    days: int = 30,
    _: str = Depends(require_admin),
):
    """Get daily revenue chart."""
    chart = get_revenue_chart(days=days)
    return {"chart": chart, "note": "Approximate — based on pro subscriptions × $19/month"}


# ── Infrastructure ─────────────────────────────────────────────────────────────

@router.get("/admin/infra/health", summary="Infrastructure health check")
async def health_endpoint(
    request: Request,
    _: str = Depends(require_admin),
):
    """Check database and API health."""
    from models.database import engine
    
    db_status = "connected" if engine else "disconnected"
    return {
        "api_status": "online",
        "db_status": db_status,
        "uptime_seconds": 0,  # TODO: track in app
        "timestamp": datetime.utcnow().isoformat(),
    }


# ── Licenses ───────────────────────────────────────────────────────────────────

@router.get("/admin/licenses", summary="List commercial licenses")
async def licenses_endpoint(
    request: Request,
    _: str = Depends(require_admin),
):
    """List all commercial licenses."""
    licenses = list_licenses()
    return {"licenses": licenses, "total": len(licenses)}


@router.post("/admin/licenses", summary="Create commercial license")
async def create_license_endpoint(
    request: Request,
    body: dict,
    _: str = Depends(require_admin),
):
    """Create a new commercial license and emit event."""
    ip = _get_client_ip(request)
    
    result = create_license(body)
    if not result:
        raise HTTPException(status_code=400, detail="Invalid license data")
    
    append_audit_log(
        action="create_license",
        target_type="license",
        target_id=result.get("id"),
        details={"company": body.get("company")},
        ip=ip,
    )
    
    await emit_event("license_created", {
        "id": result.get("id"),
        "company": result.get("company"),
        "timestamp": datetime.utcnow().isoformat(),
    })
    
    return result


# ── Audit log ──────────────────────────────────────────────────────────────────

@router.get("/admin/audit", summary="Admin audit log")
async def audit_log_endpoint(
    request: Request,
    page: int = 1,
    limit: int = 100,
    action: str = "",
    _: str = Depends(require_admin),
):
    """List audit log with pagination and filtering."""
    result = list_audit_log(page=page, limit=limit, action_filter=action)
    return result


# ── WebSocket ──────────────────────────────────────────────────────────────────

@router.websocket("/admin/ws")
async def admin_websocket(ws: WebSocket, token: str = ""):
    """
    WebSocket endpoint for live admin feed.
    Validates token on connection, closes if invalid.
    """
    if not token or not validate_admin_token(token):
        await ws.close(code=1008, reason="Unauthorized")
        return
    
    await ws.accept()
    await ws_manager.connect(ws)
    
    try:
        # Send initial snapshot
        live_job_count = len(_jobs)
        stats = get_overview_stats(live_jobs=live_job_count)
        await ws.send_json({
            "type": "connected",
            "data": stats,
            "timestamp": datetime.utcnow().isoformat(),
        })
        
        # Keep connection alive, listen for pings
        while True:
            data = await ws.receive_text()
            if data == "ping":
                await ws.send_text("pong")
    except Exception as e:
        logger.debug("WebSocket error: %s", e)
    finally:
        ws_manager.disconnect(ws)
