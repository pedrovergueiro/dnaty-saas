from prometheus_client import Counter, Histogram, Gauge
import time

# Counters
compression_total = Counter(
    'dnaty_compressions_total',
    'Total number of compression jobs',
    ['status']
)

api_requests_total = Counter(
    'dnaty_api_requests_total',
    'Total API requests',
    ['method', 'endpoint', 'status']
)

users_total = Counter(
    'dnaty_users_total',
    'Total users registered',
    ['plan']
)

# Histograms
compression_duration = Histogram(
    'dnaty_compression_duration_seconds',
    'Time spent on compression',
    buckets=[10, 30, 60, 300, 900]
)

api_request_duration = Histogram(
    'dnaty_api_request_duration_seconds',
    'API request latency',
    ['method', 'endpoint'],
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0]
)

flops_reduction = Histogram(
    'dnaty_flops_reduction_percent',
    'FLOPs reduction percentage',
    buckets=[10, 20, 30, 40, 50, 60, 70, 80, 90]
)

# Gauges
active_compressions = Gauge(
    'dnaty_active_compressions',
    'Number of ongoing compression jobs'
)

api_uptime = Gauge(
    'dnaty_api_uptime_seconds',
    'API uptime in seconds'
)

active_users = Gauge(
    'dnaty_active_users',
    'Number of active users'
)

redis_connections = Gauge(
    'dnaty_redis_connections',
    'Number of Redis connections'
)

def record_compression(duration: float, flops_pct: float, status: str = "success"):
    """Record a compression event"""
    compression_total.labels(status=status).inc()
    compression_duration.observe(duration)
    flops_reduction.observe(flops_pct)

def record_api_request(method: str, endpoint: str, duration: float, status_code: int):
    """Record an API request"""
    api_requests_total.labels(
        method=method,
        endpoint=endpoint,
        status=status_code
    ).inc()
    api_request_duration.labels(method=method, endpoint=endpoint).observe(duration)

def increment_active_compressions():
    """Increment active compression jobs"""
    active_compressions.inc()

def decrement_active_compressions():
    """Decrement active compression jobs"""
    active_compressions.dec()

def set_active_users(count: int):
    """Set number of active users"""
    active_users.set(count)

def set_api_uptime(seconds: float):
    """Set API uptime"""
    api_uptime.set(seconds)
