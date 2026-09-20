use std::os::unix::process::CommandExt;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::Mutex;

use dispatch_router::app::{build_app, AppState};
use dispatch_router::client::ModelServerClient;
use dispatch_router::metrics::RouterMetrics;
use dispatch_router::queue::AdmissionQueue;

const TEST_PORT: u16 = 50099;

struct StubServerGuard(Child);

impl Drop for StubServerGuard {
    fn drop(&mut self) {
        // `uv run python ...` forks rather than exec'ing into the real
        // interpreter on this machine -- found live: killing only the
        // immediate `uv` child left the actual model_server.py process
        // running, orphaned and reparented to launchd, still listening
        // on TEST_PORT after the test had already passed. spawn_stub_server
        // puts this child in its own process group (process_group(0)), so
        // killing the *group* (negative pid) reaches uv, python, and
        // anything else uv forked, not just the one pid this Child handle
        // names.
        let pid = self.0.id();
        let _ = Command::new("kill")
            .args(["-9", &format!("-{pid}")])
            .status();
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("router/ has a parent directory")
        .to_path_buf()
}

fn spawn_stub_server() -> StubServerGuard {
    // stderr must NOT be Stdio::inherit(): that hands this process's own
    // stderr fd to `uv run` and everything it execs. If this test's own
    // output is itself piped somewhere (`cargo test ... | tail`, as in a
    // manual invocation), the inherited fd is the pipe's write end --
    // and if this test process is then killed before its Drop guard runs
    // (SIGKILL, or the harness reclaiming a timed-out foreground command),
    // the orphaned Python subprocess keeps that fd open forever, so the
    // pipe reader never sees EOF and hangs indefinitely. Found live: a
    // manually piped `cargo test ... | tail -60` run hung for 27+ minutes
    // after this test's own process was SIGKILLed, with the model server
    // still running as an orphan reparented to launchd. Stdio::piped()
    // (captured, never inherited) closes that hole.
    let child = Command::new("uv")
        .args([
            "run",
            "python",
            "scripts/run_model_server.py",
            "--responder",
            "stub",
            "--port",
            &TEST_PORT.to_string(),
        ])
        .current_dir(repo_root())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .process_group(0)
        .spawn()
        .expect("failed to spawn scripts/run_model_server.py -- is uv on PATH?");
    StubServerGuard(child)
}

async fn wait_for_server_ready(endpoint: &str) {
    for _ in 0..50 {
        if ModelServerClient::connect(endpoint.to_string())
            .await
            .is_ok()
        {
            return;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    panic!("stub model server never became reachable at {endpoint}");
}

#[tokio::test]
async fn router_streams_real_python_stub_server_responses_over_http() {
    let _guard = spawn_stub_server();
    let endpoint = format!("http://127.0.0.1:{TEST_PORT}");
    wait_for_server_ready(&endpoint).await;

    let client = ModelServerClient::connect(endpoint)
        .await
        .expect("connect to the real Python stub server");
    let state = AppState {
        queue: Arc::new(AdmissionQueue::new(1)),
        client: Arc::new(Mutex::new(client)),
        metrics: Arc::new(RouterMetrics::new()),
    };
    let app = build_app(state);

    let request = axum::http::Request::builder()
        .method("POST")
        .uri("/generate")
        .header("content-type", "application/json")
        .body(axum::body::Body::from(
            r#"{"prompt":"hi","max_new_tokens":3}"#,
        ))
        .expect("build request");

    use tower::ServiceExt;
    let response = app.oneshot(request).await.expect("router did not panic");
    assert_eq!(response.status(), axum::http::StatusCode::OK);

    use http_body_util::BodyExt;
    let body = response
        .into_body()
        .collect()
        .await
        .expect("read body")
        .to_bytes();
    let parsed: dispatch_router::app::GenerateHttpResponse =
        serde_json::from_slice(&body).expect("valid json response");

    // StubResponder's default tokens, from model_server.py.
    assert_eq!(parsed.text, "The quick brown");
}
