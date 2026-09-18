//! HTTP endpoint speaking the System One contract: POST /v1/systemone, GET /health.
//! The scorer is not Send, so it lives on one thread and handlers queue jobs to it.

use crate::scorer::Scorer;
use crate::systemone::{Request, Response, answer};
use anyhow::Result;
use axum::{Json, Router, extract::State, http::StatusCode, routing::{get, post}};
use std::sync::{Arc, Mutex, mpsc};

type Job = (Request, mpsc::Sender<Result<Response>>);

#[derive(Clone)]
struct App {
    tx: Arc<Mutex<mpsc::Sender<Job>>>,
}

async fn systemone(State(app): State<App>, Json(req): Json<Request>) -> Result<Json<Response>, (StatusCode, String)> {
    let (rtx, rrx) = mpsc::channel();
    app.tx.lock().unwrap().send((req, rtx)).map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    let res = tokio::task::spawn_blocking(move || rrx.recv())
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    res.map(Json).map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))
}

async fn health() -> &'static str {
    "ok"
}

pub fn run(make_scorer: impl FnOnce() -> Result<Scorer> + Send + 'static, port: u16) -> Result<()> {
    let (tx, rx) = mpsc::channel::<Job>();
    let (ready_tx, ready_rx) = mpsc::channel::<Result<()>>();
    std::thread::spawn(move || {
        let mut sc = match make_scorer() {
            Ok(s) => { let _ = ready_tx.send(Ok(())); s }
            Err(e) => { let _ = ready_tx.send(Err(e)); return; }
        };
        for (req, reply) in rx {
            let _ = reply.send(answer(&mut sc, &req));
        }
    });
    ready_rx.recv()??;
    let app = Router::new()
        .route("/v1/systemone", post(systemone))
        .route("/health", get(health))
        .with_state(App { tx: Arc::new(Mutex::new(tx)) });
    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(async {
        let listener = tokio::net::TcpListener::bind(("127.0.0.1", port)).await?;
        eprintln!("gliner2-doom serve: http://127.0.0.1:{port}/v1/systemone  (GET /health)");
        axum::serve(listener, app).await?;
        Ok::<(), anyhow::Error>(())
    })
}
