import time
import random
from http.server import HTTPServer, BaseHTTPRequestHandler

# In a true enterprise setup, this script scrapes the C++ Watchdog output
# and the physical Linux /sys/class/thermal sensors, formatting them for Prometheus.

class TelemetryExporter(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/metrics':
            self.send_response(200)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            
            # Simulated metrics (Normally read from IPC or /sys/class)
            pseudo_temp = random.uniform(55.0, 78.0)
            pseudo_fps = 30.0 if pseudo_temp < 75.0 else 15.0
            memory_usage_mb = random.uniform(160.0, 210.0)
            watchdog_resets = 0

            # Prometheus text-based exposition format
            metrics = f"""
            # HELP cow_bcs_pipeline_fps Current pipeline throughput in Frames Per Second
            # TYPE cow_bcs_pipeline_fps gauge
            cow_bcs_pipeline_fps {pseudo_fps}
            
            # HELP cow_bcs_soc_temperature_celsius SoC physical thermal sensor in Celsius
            # TYPE cow_bcs_soc_temperature_celsius gauge
            cow_bcs_soc_temperature_celsius {pseudo_temp:.2f}
            
            # HELP cow_bcs_memory_rss_mb System RAM consumption in Megabytes
            # TYPE cow_bcs_memory_rss_mb gauge
            cow_bcs_memory_rss_mb {memory_usage_mb:.2f}
            
            # HELP cow_bcs_watchdog_resets_total Number of times the RTSP hardware decoder watchdog flushed the pipeline
            # TYPE cow_bcs_watchdog_resets_total counter
            cow_bcs_watchdog_resets_total {watchdog_resets}
            """
            
            self.wfile.write(metrics.encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()

def run(server_class=HTTPServer, handler_class=TelemetryExporter, port=8000):
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    print(f"[Telemetry] Prometheus Exporter listening on port {port}...")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    httpd.server_close()
    print("[Telemetry] Exporter stopped.")

if __name__ == '__main__':
    run()
