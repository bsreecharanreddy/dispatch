use std::env;
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::Mutex;

use dispatch_router::app::{build_app, AppState};
use dispatch_router::client::ModelServerClient;
use dispatch_router::metrics::RouterMetrics;
use dispatch_router::queue::AdmissionQueue;

const CONNECT_RETRY_ATTEMPTS: u32 = 30;
const CONNECT_RETRY_DELAY: Duration = Duration::from_secs(1);

/// Container/pod startup order guarantees the model server's *container*
/// has started, not that it has finished syncing its venv and bound its
/// port -- found live: a docker-compose run had the router exit
/// immediately on a single failed connect attempt while the model server
/// was still starting. Retries for up to CONNECT_RETRY_ATTEMPTS *
/// CONNECT_RETRY_DELAY before giving up for good.
async fn connect_with_retry(endpoint: &str) -> Result<ModelServerClient, tonic::transport::Error> {
    let mut last_err = None;
    for attempt in 1..=CONNECT_RETRY_ATTEMPTS {
        match ModelServerClient::connect(endpoint.to_string()).await {
            Ok(client) => return Ok(client),
            Err(err) => {
                println!(
                    "model server not ready yet (attempt {attempt}/{CONNECT_RETRY_ATTEMPTS}): {err}"
                );
                last_err = Some(err);
                tokio::time::sleep(CONNECT_RETRY_DELAY).await;
            }
        }
    }
    Err(last_err.expect("loop runs at least once"))
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_server_endpoint =
        env::var("MODEL_SERVER_ENDPOINT").unwrap_or_else(|_| "http://127.0.0.1:50051".to_string());
    let listen_addr = env::var("ROUTER_LISTEN_ADDR").unwrap_or_else(|_| "0.0.0.0:8080".to_string());
    let queue_capacity: usize = env::var("ROUTER_QUEUE_CAPACITY")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1);

    println!("connecting to model server at {model_server_endpoint}");
    let client = connect_with_retry(&model_server_endpoint).await?;

    let state = AppState {
        queue: Arc::new(AdmissionQueue::new(queue_capacity)),
        client: Arc::new(Mutex::new(client)),
        metrics: Arc::new(RouterMetrics::new()),
    };

    let app = build_app(state);
    let listener = tokio::net::TcpListener::bind(&listen_addr).await?;
    println!("dispatch-router listening on {listen_addr}");
    axum::serve(listener, app).await?;
    Ok(())
}
