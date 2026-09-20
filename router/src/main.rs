use std::env;
use std::sync::Arc;

use tokio::sync::Mutex;

use dispatch_router::app::{build_app, AppState};
use dispatch_router::client::ModelServerClient;
use dispatch_router::metrics::RouterMetrics;
use dispatch_router::queue::AdmissionQueue;

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
    let client = ModelServerClient::connect(model_server_endpoint).await?;

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
