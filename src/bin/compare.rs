use anyhow::{Context, Result, ensure};
use clap::Parser;
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
};

/// Compare like-for-like synthetic baseline runs. Does not establish real-world quality.
#[derive(Parser)]
struct Args {
    baseline: PathBuf,
    candidate: PathBuf,
    /// Permit an explicitly patched backend; model, inputs and work must still match.
    #[arg(long)]
    allow_backend_change: bool,
}
fn read(dir: &Path, file: &str) -> Result<Value> {
    Ok(serde_json::from_slice(&fs::read(dir.join(file))?)?)
}
fn same_contract(a: &Value, b: &Value, allow_backend_change: bool) -> Result<()> {
    ensure!(
        !a["profiling"].as_bool().unwrap_or(false) && !b["profiling"].as_bool().unwrap_or(false),
        "instrumented runs are not performance comparisons"
    );
    for key in [
        "schema",
        "backend",
        "media_tools",
        "model_sha256",
        "mmproj_sha256",
        "repeats",
        "input",
    ] {
        ensure!(!a[key].is_null() && a[key] == b[key], "incompatible {key}");
    }
    for key in ["server_sha256", "backend_libraries"] {
        ensure!(!a[key].is_null() && !b[key].is_null(), "missing {key}");
        ensure!(
            allow_backend_change || a[key] == b[key],
            "incompatible {key}"
        );
    }
    if allow_backend_change {
        for key in ["batch", "ubatch"] {
            ensure!(
                !a["settings"][key].is_null() && a["settings"][key] == b["settings"][key],
                "backend experiment also changed {key}"
            );
        }
    }
    for key in [
        "context",
        "threads",
        "graphs",
        "flash_attention",
        "cache_prompt",
        "parallel",
        "video_fps",
    ] {
        ensure!(
            !a["settings"][key].is_null() && a["settings"][key] == b["settings"][key],
            "incompatible setting {key}"
        );
    }
    ensure!(
        a["video_sha256"] == b["video_sha256"],
        "video fixtures differ"
    );
    Ok(())
}
fn samples(dir: &Path) -> Result<BTreeMap<String, Vec<Value>>> {
    let mut result = BTreeMap::<String, Vec<Value>>::new();
    for line in fs::read_to_string(dir.join("samples.jsonl"))?.lines() {
        let v: Value = serde_json::from_str(line)?;
        if v["warmup"] == false {
            result
                .entry(v["case"].as_str().context("missing case name")?.to_owned())
                .or_default()
                .push(v);
        }
    }
    Ok(result)
}
fn compare(a_dir: &Path, b_dir: &Path, allow_backend_change: bool) -> Result<Value> {
    let a_meta = read(a_dir, "metadata.json")?;
    let b_meta = read(b_dir, "metadata.json")?;
    same_contract(&a_meta, &b_meta, allow_backend_change)?;
    // Require completion markers: a partial run must never become a result.
    let a_summary = read(a_dir, "summary.json")?;
    let b_summary = read(b_dir, "summary.json")?;
    let a = samples(a_dir)?;
    let b = samples(b_dir)?;
    ensure!(
        a.keys().eq(b.keys()) && !a.is_empty(),
        "workload sets differ or are empty"
    );
    let mut cases = Vec::new();
    for (count, left) in &a {
        let right = &b[count];
        let repeats = a_meta["repeats"].as_u64().context("missing repeats")? as usize;
        ensure!(
            left.len() == repeats && right.len() == repeats,
            "incomplete sample count"
        );
        let mut outputs_equal = true;
        let mut a_total = 0.;
        let mut b_total = 0.;
        for (x, y) in left.iter().zip(right) {
            ensure!(
                x["request_sha256"].is_string() && x["request_sha256"] == y["request_sha256"],
                "input mismatch"
            );
            for z in [x, y] {
                ensure!(z["response"]["timings"]["cache_n"] == 0, "cached prompt");
            }
            for key in ["prompt_n", "predicted_n"] {
                ensure!(
                    x["response"]["timings"][key].as_u64().unwrap_or(0) > 0
                        && x["response"]["timings"][key] == y["response"]["timings"][key],
                    "different evaluated work: {key}"
                );
            }
            outputs_equal &= x["response"]["choices"][0]["message"]["content"]
                == y["response"]["choices"][0]["message"]["content"];
            let at = x["wall_ms"].as_f64().context("missing wall time")?;
            let bt = y["wall_ms"].as_f64().context("missing wall time")?;
            ensure!(
                at > 0. && bt > 0. && at.is_finite() && bt.is_finite(),
                "invalid wall time"
            );
            a_total += at;
            b_total += bt;
        }
        cases.push(json!({"case":count,"baseline_mean_ms":a_total/repeats as f64,"candidate_mean_ms":b_total/repeats as f64,"speedup":a_total/b_total,"sample_outputs_equal":outputs_equal,"synthetic_gate_passes":outputs_equal && a_total/b_total >= 1.15}));
    }
    Ok(
        json!({"baseline":a_summary["label"],"candidate":b_summary["label"],"cases":cases,"allow_backend_change":allow_backend_change,"baseline_backend":{"server":a_meta["server_sha256"],"libraries":a_meta["backend_libraries"]},"candidate_backend":{"server":b_meta["server_sha256"],"libraries":b_meta["backend_libraries"]},"scope":"matched model, inputs and evaluated work; synthetic output agreement is not general quality parity; inspect hardware and thermal conditions separately"}),
    )
}
fn main() -> Result<()> {
    let args = Args::parse();
    println!(
        "{}",
        serde_json::to_string_pretty(&compare(
            &args.baseline,
            &args.candidate,
            args.allow_backend_change
        )?)?
    );
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rejects_model_and_graph_policy_changes() {
        let a = json!({"schema":2,"backend":"llama.cpp","server_sha256":"a","backend_libraries":{"libllama.so":"d"},"media_tools":{},"model_sha256":"b","mmproj_sha256":"c","repeats":5,"input":"synthetic","settings":{"context":1024,"threads":6,"graphs":true,"flash_attention":true,"cache_prompt":false,"parallel":1,"ubatch":512,"video_fps":4}});
        let mut b = a.clone();
        b["settings"]["ubatch"] = json!(1024);
        assert!(same_contract(&a, &b, false).is_ok());
        b["model_sha256"] = json!("different");
        assert!(same_contract(&a, &b, false).is_err());
        assert!(same_contract(&a, &b, true).is_err());
        b = a.clone();
        b["settings"]["graphs"] = json!(false);
        assert!(same_contract(&a, &b, false).is_err());
        b = a.clone();
        b["backend_libraries"]["libllama.so"] = json!("changed");
        assert!(same_contract(&a, &b, false).is_err());
        let mut a = a;
        a["settings"]["batch"] = json!(2048);
        b["settings"]["batch"] = json!(2048);
        assert!(same_contract(&a, &b, true).is_ok());
        b["profiling"] = json!(true);
        assert!(same_contract(&a, &b, true).is_err());
    }
}
