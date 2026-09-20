use std::time::{Duration, Instant};

use tonic::transport::Channel;
use tonic::Request;

use crate::metrics::RouterMetrics;
use crate::pb::dispatch_v1::model_server_client::ModelServerClient as GrpcClient;
use crate::pb::dispatch_v1::GenerateRequest;

pub struct ModelServerClient {
    inner: GrpcClient<Channel>,
}

#[derive(Debug, Clone)]
pub struct StreamedGeneration {
    pub full_text: String,
    pub ttft: Duration,
    pub inter_token_latencies: Vec<Duration>,
}

impl ModelServerClient {
    pub async fn connect(endpoint: String) -> Result<Self, tonic::transport::Error> {
        let inner = GrpcClient::connect(endpoint).await?;
        Ok(Self { inner })
    }

    pub async fn generate(
        &mut self,
        request_id: &str,
        prompt: &str,
        max_new_tokens: u32,
        metrics: &RouterMetrics,
    ) -> Result<StreamedGeneration, tonic::Status> {
        let start = Instant::now();
        let mut stream = self
            .inner
            .generate(Request::new(GenerateRequest {
                request_id: request_id.to_string(),
                prompt: prompt.to_string(),
                max_new_tokens,
            }))
            .await?
            .into_inner();

        let mut full_text = String::new();
        let mut ttft: Option<Duration> = None;
        let mut inter_token_latencies = Vec::new();
        let mut last_token_at = start;

        while let Some(chunk) = stream.message().await? {
            if chunk.is_final {
                break;
            }
            let now = Instant::now();
            match ttft {
                None => {
                    let elapsed = now.duration_since(start);
                    metrics.ttft_seconds.observe(elapsed.as_secs_f64());
                    ttft = Some(elapsed);
                }
                Some(_) => {
                    let gap = now.duration_since(last_token_at);
                    metrics
                        .inter_token_latency_seconds
                        .observe(gap.as_secs_f64());
                    inter_token_latencies.push(gap);
                }
            }
            last_token_at = now;
            full_text.push_str(&chunk.text);
        }

        Ok(StreamedGeneration {
            full_text,
            ttft: ttft.unwrap_or_default(),
            inter_token_latencies,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::spawn_mock_model_server;

    #[tokio::test]
    async fn generate_accumulates_text_and_records_latencies() {
        let endpoint = spawn_mock_model_server(vec!["Hello", " world", "!"]).await;
        let mut client = ModelServerClient::connect(endpoint).await.expect("connect");
        let metrics = RouterMetrics::new();

        let result = client
            .generate("req-1", "hi", 3, &metrics)
            .await
            .expect("generate succeeds");

        assert_eq!(result.full_text, "Hello world!");
        assert_eq!(result.inter_token_latencies.len(), 2);
        assert!(metrics
            .encode()
            .contains("dispatch_router_ttft_seconds_count 1"));
    }
}
