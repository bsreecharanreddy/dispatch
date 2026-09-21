use std::pin::Pin;

use tokio::net::TcpListener;
use tokio::sync::mpsc;
use tokio_stream::wrappers::{ReceiverStream, TcpListenerStream};
use tonic::{Request, Response, Status};

use crate::pb::dispatch_v1::model_server_server::{ModelServer, ModelServerServer};
use crate::pb::dispatch_v1::{GenerateRequest, GenerateResponse};

struct MockModelServer {
    chunks: Vec<&'static str>,
}

#[tonic::async_trait]
impl ModelServer for MockModelServer {
    type GenerateStream =
        Pin<Box<dyn futures_core::Stream<Item = Result<GenerateResponse, Status>> + Send>>;

    async fn generate(
        &self,
        request: Request<GenerateRequest>,
    ) -> Result<Response<Self::GenerateStream>, Status> {
        let request_id = request.into_inner().request_id;
        let chunks = self.chunks.clone();
        let (tx, rx) = mpsc::channel(8);
        tokio::spawn(async move {
            for text in chunks {
                tokio::time::sleep(std::time::Duration::from_millis(5)).await;
                let _ = tx
                    .send(Ok(GenerateResponse {
                        request_id: request_id.clone(),
                        text: text.to_string(),
                        is_final: false,
                        t_emit_unix: 0.0,
                    }))
                    .await;
            }
            let _ = tx
                .send(Ok(GenerateResponse {
                    request_id,
                    text: String::new(),
                    is_final: true,
                    t_emit_unix: 0.0,
                }))
                .await;
        });
        Ok(Response::new(Box::pin(ReceiverStream::new(rx))))
    }
}

/// Starts an in-process, real gRPC server implementing ModelServer that
/// streams `chunks` back as separate non-final tokens followed by one
/// final empty message, then returns its `http://host:port` endpoint.
/// Test-only (router/src/lib.rs gates this module behind `#[cfg(test)]`).
pub async fn spawn_mock_model_server(chunks: Vec<&'static str>) -> String {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind ephemeral port");
    let addr = listener.local_addr().expect("local addr");
    tokio::spawn(async move {
        tonic::transport::Server::builder()
            .add_service(ModelServerServer::new(MockModelServer { chunks }))
            .serve_with_incoming(TcpListenerStream::new(listener))
            .await
            .expect("mock server exited unexpectedly");
    });
    tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    format!("http://{addr}")
}
