use std::sync::Arc;

use axum::extract::State;
use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;

use crate::client::ModelServerClient;
use crate::metrics::RouterMetrics;
use crate::queue::AdmissionQueue;

#[derive(Clone)]
pub struct AppState {
    pub queue: Arc<AdmissionQueue>,
    pub client: Arc<Mutex<ModelServerClient>>,
    pub metrics: Arc<RouterMetrics>,
}

#[derive(Debug, Deserialize)]
pub struct GenerateHttpRequest {
    pub prompt: String,
    pub max_new_tokens: u32,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct GenerateHttpResponse {
    pub text: String,
    pub ttft_ms: f64,
    pub inter_token_latencies_ms: Vec<f64>,
}

pub fn build_app(state: AppState) -> Router {
    Router::new()
        .route("/generate", post(generate))
        .route("/healthz", get(healthz))
        .route("/readyz", get(readyz))
        .route("/metrics", get(metrics_handler))
        .with_state(state)
}

async fn generate(
    State(state): State<AppState>,
    Json(req): Json<GenerateHttpRequest>,
) -> Result<Json<GenerateHttpResponse>, axum::http::StatusCode> {
    let _ticket = state.queue.admit().await;
    state.metrics.queue_depth.set(state.queue.depth() as i64);
    state.metrics.requests_total.inc();

    let mut client = state.client.lock().await;
    let request_id = request_id();
    let result = client
        .generate(&request_id, &req.prompt, req.max_new_tokens, &state.metrics)
        .await
        .map_err(|_| axum::http::StatusCode::BAD_GATEWAY)?;

    Ok(Json(GenerateHttpResponse {
        text: result.full_text,
        ttft_ms: result.ttft.as_secs_f64() * 1000.0,
        inter_token_latencies_ms: result
            .inter_token_latencies
            .iter()
            .map(|d| d.as_secs_f64() * 1000.0)
            .collect(),
    }))
}

async fn healthz() -> &'static str {
    "ok"
}

/// Startup blocks on a successful connection to the model server (see
/// main.rs) -- by the time this process is serving HTTP at all, it is
/// also ready, so readyz and healthz report the same thing here. Kept
/// as a separate endpoint because K8s conventionally wires liveness and
/// readiness probes to different paths.
async fn readyz() -> &'static str {
    "ok"
}

async fn metrics_handler(State(state): State<AppState>) -> impl IntoResponse {
    state.metrics.encode()
}

fn request_id() -> String {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("system clock is after the epoch")
        .as_nanos();
    format!("{nanos:x}")
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use http_body_util::BodyExt;
    use tokio::sync::Mutex;
    use tower::ServiceExt;

    use super::*;
    use crate::client::ModelServerClient;
    use crate::metrics::RouterMetrics;
    use crate::queue::AdmissionQueue;
    use crate::test_support::spawn_mock_model_server;

    async fn test_state(chunks: Vec<&'static str>) -> AppState {
        let endpoint = spawn_mock_model_server(chunks).await;
        let client = ModelServerClient::connect(endpoint).await.expect("connect");
        AppState {
            queue: Arc::new(AdmissionQueue::new(1)),
            client: Arc::new(Mutex::new(client)),
            metrics: Arc::new(RouterMetrics::new()),
        }
    }

    #[tokio::test]
    async fn generate_endpoint_returns_full_text_and_timing() {
        let app = build_app(test_state(vec!["Hello", " world"]).await);

        let request = axum::http::Request::builder()
            .method("POST")
            .uri("/generate")
            .header("content-type", "application/json")
            .body(axum::body::Body::from(
                r#"{"prompt":"hi","max_new_tokens":2}"#,
            ))
            .expect("build request");

        let response = app.oneshot(request).await.expect("router did not panic");
        assert_eq!(response.status(), axum::http::StatusCode::OK);

        let body = response
            .into_body()
            .collect()
            .await
            .expect("read body")
            .to_bytes();
        let parsed: GenerateHttpResponse =
            serde_json::from_slice(&body).expect("valid json response");
        assert_eq!(parsed.text, "Hello world");
        assert!(parsed.ttft_ms >= 0.0);
    }

    #[tokio::test]
    async fn healthz_and_readyz_return_ok() {
        for path in ["/healthz", "/readyz"] {
            let app = build_app(test_state(vec!["ok"]).await);
            let request = axum::http::Request::builder()
                .uri(path)
                .body(axum::body::Body::empty())
                .expect("build request");
            let response = app.oneshot(request).await.expect("router did not panic");
            assert_eq!(
                response.status(),
                axum::http::StatusCode::OK,
                "path: {path}"
            );
        }
    }

    #[tokio::test]
    async fn metrics_endpoint_exposes_prometheus_text_format() {
        let app = build_app(test_state(vec!["ok"]).await);

        let request = axum::http::Request::builder()
            .uri("/metrics")
            .body(axum::body::Body::empty())
            .expect("build request");
        let response = app.oneshot(request).await.expect("router did not panic");
        assert_eq!(response.status(), axum::http::StatusCode::OK);

        let body = response
            .into_body()
            .collect()
            .await
            .expect("read body")
            .to_bytes();
        let text = String::from_utf8(body.to_vec()).expect("utf8");
        assert!(text.contains("dispatch_router_queue_depth"));
    }
}
