//! TypeSafe "System One" request/response shape, answered by GLiNER2.5.
//!
//! One `state` plus a map of typed questions. Each question is one
//! single-label classification over the state text (instructions prepended),
//! so a question costs one encoder pass; nothing is generated.
//!
//! choice: criteria = { option: description | null }  -> choice, probabilities, confidence
//! score:  criteria = [ level description, ... ]      -> score (expected level index), probabilities, legend
//! noul:   criteria = { "true": .., "false": .. }?    -> noul = P(yes)

use crate::scorer::Scorer;
use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use std::time::Instant;

#[derive(Debug, Deserialize)]
pub struct Request {
    pub state: Value,
    #[serde(default)]
    pub model: Option<String>,
    pub questions: Map<String, Value>,
}

#[derive(Debug, Serialize)]
pub struct Response {
    pub model: String,
    pub answers: Map<String, Value>,
    pub usage: Value,
    pub timing_ms: f64,
}

fn state_text(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        other => serde_json::to_string_pretty(other).unwrap_or_default(),
    }
}

fn confidence(probs: &[f32]) -> f32 {
    let n = probs.len().max(2) as f32;
    let h: f32 = probs.iter().filter(|p| **p > 0.0).map(|p| -p * p.ln()).sum();
    (1.0 - h / n.ln()).clamp(0.0, 1.0)
}

/// The state is classified as-is. Measured 2026-09-17: prepending the
/// instructions to the text made every message read as "furious" and broke
/// urgency; the same labels over the bare state answered sensibly. The
/// instructions survive only as the default yes-label of a `noul` question.
fn text_for(_instructions: Option<&str>, state: &str) -> String {
    state.to_string()
}

/// Runs one question. Returns the answer object.
fn answer_one(sc: &mut Scorer, qid: &str, q: &Value, state: &str) -> Result<Value> {
    let qtype = q.get("type").and_then(Value::as_str).ok_or_else(|| anyhow!("question {qid}: missing type"))?;
    let instructions = q.get("instructions").and_then(Value::as_str);
    let text = text_for(instructions, state);
    match qtype {
        "choice" => {
            let crit = q.get("criteria").and_then(Value::as_object).ok_or_else(|| anyhow!("question {qid}: choice needs criteria object"))?;
            let names: Vec<String> = crit.keys().cloned().collect();
            let labels: Vec<String> = crit
                .iter()
                .map(|(k, v)| match v {
                    Value::String(d) if !d.is_empty() => format!("{k}: {d}"),
                    Value::Null => k.clone(),
                    other => format!("{k}: {}", serde_json::to_string(other).unwrap_or_default()),
                })
                .collect();
            let scored = sc.classify(qid, &text, &labels)?;
            let mut probs = Map::new();
            let mut best = (String::new(), -1.0f32);
            let mut pvec = vec![];
            for s in &scored {
                let idx = labels.iter().position(|l| *l == s.label).unwrap_or(0);
                probs.insert(names[idx].clone(), json!(s.prob));
                pvec.push(s.prob);
                if s.prob > best.1 {
                    best = (names[idx].clone(), s.prob);
                }
            }
            Ok(json!({"choice": best.0, "probabilities": probs, "confidence": confidence(&pvec)}))
        }
        "score" => {
            let levels = q.get("criteria").and_then(Value::as_array).ok_or_else(|| anyhow!("question {qid}: score needs criteria array"))?;
            let labels: Vec<String> = levels.iter().map(|v| v.as_str().map(str::to_string).unwrap_or_else(|| v.to_string())).collect();
            let scored = sc.classify(qid, &text, &labels)?;
            let mut probs = Map::new();
            let mut pvec = vec![0f32; labels.len()];
            for s in &scored {
                let idx = labels.iter().position(|l| *l == s.label).unwrap_or(0);
                pvec[idx] = s.prob;
                probs.insert(idx.to_string(), json!(s.prob));
            }
            let score: f32 = pvec.iter().enumerate().map(|(i, p)| i as f32 * p).sum();
            Ok(json!({"score": score, "probabilities": probs, "legend": labels, "confidence": confidence(&pvec)}))
        }
        "noul" => {
            let crit = q.get("criteria").and_then(Value::as_object);
            let desc = |k: &str| crit.and_then(|c| c.get(k)).and_then(Value::as_str).filter(|s| !s.is_empty());
            // Bare yes/no carries nothing for a label matcher: the yes label is
            // the statement being asked (criteria.true, else the instructions).
            let yes_default = instructions.map(|i| i.trim().trim_end_matches('?').to_string()).unwrap_or_else(|| "yes".into());
            let labels = vec![
                desc("true").map(str::to_string).unwrap_or(yes_default),
                desc("false").map(str::to_string).unwrap_or_else(|| "no".into()),
            ];
            let scored = sc.classify(qid, &text, &labels)?;
            let p_yes = scored.iter().find(|s| s.label == labels[0]).map(|s| s.prob).unwrap_or(0.0);
            Ok(json!({"noul": p_yes, "confidence": confidence(&[p_yes, 1.0 - p_yes])}))
        }
        other => Err(anyhow!("question {qid}: unknown type {other}")),
    }
}

pub fn answer(sc: &mut Scorer, req: &Request) -> Result<Response> {
    let t0 = Instant::now();
    let state = state_text(&req.state);
    let mut answers = Map::new();
    for (qid, q) in &req.questions {
        answers.insert(qid.clone(), answer_one(sc, qid, q, &state)?);
    }
    Ok(Response {
        model: req.model.clone().unwrap_or_else(|| "gliner2.5-multi-v1".into()),
        answers,
        usage: json!({"input_tokens": 0, "output_tokens": 0, "passes": req.questions.len()}),
        timing_ms: t0.elapsed().as_secs_f64() * 1000.0,
    })
}

/// Built-in examples: the TypeSafe quick-start ticket and two routing cases.
pub fn examples() -> Vec<Request> {
    let mk = |state: &str| Request {
        state: json!(state),
        model: None,
        questions: json!({
            "department": {"type": "choice", "instructions": "Which department should handle this customer message?",
                "criteria": {"technical": "Product bugs, API and integration failures", "billing": "Charges, invoices and refunds",
                             "shipping": "Delivery and lost packages", "returns": "Returns and exchanges"}},
            "frustration": {"type": "score", "instructions": "How frustrated is the customer?",
                "criteria": ["calm", "annoyed but polite", "furious"]},
            "is_urgent": {"type": "noul", "instructions": "Does this need a response within the hour?",
                "criteria": {"true": "needs a response within the hour", "false": "no"}}
        }).as_object().cloned().unwrap(),
    };
    vec![
        mk("Our Stripe integration has been failing for three days now. Every checkout returns a 500 from your API and we are losing orders. We need this fixed today."),
        mk("I ordered size 10 shoes but received size 8. Please send the right size."),
        mk("Hi, just checking whether the invoice from last month can be reissued with our new company address. No rush."),
    ]
}
