use prometheus::{
    Encoder, Histogram, HistogramOpts, IntCounter, IntGauge, Opts, Registry, TextEncoder,
};

pub struct RouterMetrics {
    registry: Registry,
    pub requests_total: IntCounter,
    pub ttft_seconds: Histogram,
    pub inter_token_latency_seconds: Histogram,
    pub queue_depth: IntGauge,
}

impl RouterMetrics {
    pub fn new() -> Self {
        let registry = Registry::new();

        let requests_total = IntCounter::with_opts(Opts::new(
            "dispatch_router_requests_total",
            "Total requests admitted by the router",
        ))
        .expect("valid metric opts");
        registry
            .register(Box::new(requests_total.clone()))
            .expect("first registration of this metric");

        let ttft_seconds = Histogram::with_opts(HistogramOpts::new(
            "dispatch_router_ttft_seconds",
            "Time to first token, seconds",
        ))
        .expect("valid metric opts");
        registry
            .register(Box::new(ttft_seconds.clone()))
            .expect("first registration of this metric");

        let inter_token_latency_seconds = Histogram::with_opts(HistogramOpts::new(
            "dispatch_router_inter_token_latency_seconds",
            "Gap between consecutive tokens, seconds",
        ))
        .expect("valid metric opts");
        registry
            .register(Box::new(inter_token_latency_seconds.clone()))
            .expect("first registration of this metric");

        let queue_depth = IntGauge::with_opts(Opts::new(
            "dispatch_router_queue_depth",
            "Requests currently waiting for the single-flight model server",
        ))
        .expect("valid metric opts");
        registry
            .register(Box::new(queue_depth.clone()))
            .expect("first registration of this metric");

        Self {
            registry,
            requests_total,
            ttft_seconds,
            inter_token_latency_seconds,
            queue_depth,
        }
    }

    pub fn encode(&self) -> String {
        let metric_families = self.registry.gather();
        let mut buffer = Vec::new();
        TextEncoder::new()
            .encode(&metric_families, &mut buffer)
            .expect("prometheus text encoding never fails for valid metric families");
        String::from_utf8(buffer).expect("prometheus TextEncoder always emits valid utf-8")
    }
}

impl Default for RouterMetrics {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_reports_recorded_values() {
        let metrics = RouterMetrics::new();
        metrics.requests_total.inc();
        metrics.ttft_seconds.observe(0.25);
        metrics.queue_depth.set(3);

        let text = metrics.encode();

        assert!(text.contains("dispatch_router_requests_total 1"));
        assert!(text.contains("dispatch_router_queue_depth 3"));
        assert!(text.contains("dispatch_router_ttft_seconds_sum 0.25"));
    }
}
