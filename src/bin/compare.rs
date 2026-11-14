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
    /// Compare CPU and native-backend execution of the same sampler on the same binary.
    #[arg(long, conflicts_with = "allow_backend_change")]
    allow_sampling_change: bool,
    /// Report answer differences even when generated lengths differ; do not claim speedups.
    #[arg(long)]
    quality_only: bool,
    /// Explicit case subset; all requested cases must exist in both complete runs.
    #[arg(long = "cases", value_delimiter = ',')]
    selected_cases: Vec<String>,
}
fn read(dir: &Path, file: &str) -> Result<Value> {
    Ok(serde_json::from_slice(&fs::read(dir.join(file))?)?)
}
fn same_contract(
    a: &Value,
    b: &Value,
    allow_backend_change: bool,
    allow_sampling_change: bool,
) -> Result<()> {
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
    let sampling = |meta: &Value| -> Result<bool> {
        // Metadata written before this option existed always used CPU sampling.
        match meta["settings"].get("backend_sampling") {
            None => Ok(false),
            Some(value) => value.as_bool().context("invalid backend_sampling setting"),
        }
    };
    let (a_sampling, b_sampling) = (sampling(a)?, sampling(b)?);
    ensure!(
        allow_sampling_change || a_sampling == b_sampling,
        "sampling backend differs"
    );
    if allow_backend_change || allow_sampling_change {
        for key in ["batch", "ubatch"] {
            ensure!(
                !a["settings"][key].is_null() && a["settings"][key] == b["settings"][key],
                "controlled experiment also changed {key}"
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
    ensure!(a["corpus"] == b["corpus"], "media corpora differ");
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
fn compare(
    a_dir: &Path,
    b_dir: &Path,
    allow_backend_change: bool,
    allow_sampling_change: bool,
    quality_only: bool,
    selected_cases: &[String],
) -> Result<Value> {
    let a_meta = read(a_dir, "metadata.json")?;
    let b_meta = read(b_dir, "metadata.json")?;
    let token_logprobs = |meta: &Value| -> Result<bool> {
        match meta.get("token_logprobs") {
            None => Ok(false),
            Some(value) => value.as_bool().context("invalid token_logprobs setting"),
        }
    };
    ensure!(
        token_logprobs(&a_meta)? == token_logprobs(&b_meta)?,
        "token probability settings differ"
    );
    ensure!(
        quality_only || !token_logprobs(&a_meta)?,
        "token-logprob runs are diagnostic, not performance evidence"
    );
    same_contract(
        &a_meta,
        &b_meta,
        allow_backend_change,
        allow_sampling_change,
    )?;
    // Require completion markers: a partial run must never become a result.
    let a_summary = read(a_dir, "summary.json")?;
    let b_summary = read(b_dir, "summary.json")?;
    let a = samples(a_dir)?;
    let b = samples(b_dir)?;
    ensure!(
        a.keys().eq(b.keys()) && !a.is_empty(),
        "workload sets differ or are empty"
    );
    ensure!(
        selected_cases.iter().all(|name| a.contains_key(name)),
        "selected case missing"
    );
    let mut cases = Vec::new();
    for (count, left) in &a {
        let right = &b[count];
        let repeats = a_meta["repeats"].as_u64().context("missing repeats")? as usize;
        ensure!(
            left.len() == repeats && right.len() == repeats,
            "incomplete sample count"
        );
        if !selected_cases.is_empty() && !selected_cases.contains(count) {
            continue;
        }
        let mut outputs_equal = true;
        let mut generated_work_equal = true;
        let mut a_total = 0.;
        let mut b_total = 0.;
        let mut token_differences = Vec::new();
        for (x, y) in left.iter().zip(right) {
            ensure!(
                x["request_sha256"].is_string() && x["request_sha256"] == y["request_sha256"],
                "input mismatch"
            );
            for z in [x, y] {
                ensure!(z["response"]["timings"]["cache_n"] == 0, "cached prompt");
            }
            for key in ["prompt_n", "predicted_n"] {
                let same = x["response"]["timings"][key] == y["response"]["timings"][key];
                if key == "predicted_n" {
                    generated_work_equal &= same;
                }
                ensure!(
                    x["response"]["timings"][key].as_u64().unwrap_or(0) > 0
                        && y["response"]["timings"][key].as_u64().unwrap_or(0) > 0
                        && (same || (quality_only && key == "predicted_n")),
                    "different evaluated work: {key}"
                );
            }
            outputs_equal &= x["response"]["choices"][0]["message"]["content"]
                == y["response"]["choices"][0]["message"]["content"];
            if token_logprobs(&a_meta)? {
                let records = |sample: &Value| -> Result<Vec<Value>> {
                    let tokens = sample["response"]["choices"][0]["logprobs"]["content"]
                        .as_array()
                        .context("missing diagnostic token probabilities")?;
                    ensure!(
                        !tokens.is_empty()
                            && tokens.iter().all(|t| t["id"].is_u64()
                                && t["token"].is_string()
                                && t["logprob"].as_f64().is_some_and(f64::is_finite)
                                && t["top_logprobs"].as_array().is_some_and(|v| !v.is_empty())),
                        "invalid diagnostic token probabilities"
                    );
                    Ok(tokens.clone())
                };
                let xt = records(x)?;
                let yt = records(y)?;
                let common = xt
                    .iter()
                    .zip(&yt)
                    .take_while(|(a, b)| a["id"] == b["id"])
                    .count();
                token_differences.push(json!({
                    "common_prefix_tokens":common,
                    "first_difference_index":if common == xt.len() && common == yt.len() {None} else {Some(common)},
                    "baseline":xt.get(common),"candidate":yt.get(common)
                }));
            }
            let at = x["wall_ms"].as_f64().context("missing wall time")?;
            let bt = y["wall_ms"].as_f64().context("missing wall time")?;
            ensure!(
                at > 0. && bt > 0. && at.is_finite() && bt.is_finite(),
                "invalid wall time"
            );
            a_total += at;
            b_total += bt;
        }
        let mut result = json!({"case":count,"sample_outputs_equal":outputs_equal,"generated_work_equal":generated_work_equal});
        if quality_only {
            if token_logprobs(&a_meta)? {
                result["token_differences"] = json!(token_differences);
            }
            result["baseline_outputs"] = json!(
                left.iter()
                    .map(|v| &v["response"]["choices"][0]["message"]["content"])
                    .collect::<Vec<_>>()
            );
            result["candidate_outputs"] = json!(
                right
                    .iter()
                    .map(|v| &v["response"]["choices"][0]["message"]["content"])
                    .collect::<Vec<_>>()
            );
        } else {
            result["baseline_mean_ms"] = json!(a_total / repeats as f64);
            result["candidate_mean_ms"] = json!(b_total / repeats as f64);
            result["speedup"] = json!(a_total / b_total);
            result["synthetic_gate_passes"] = json!(outputs_equal && a_total / b_total >= 1.15);
        }
        cases.push(result);
    }
    Ok(
        json!({"baseline":a_summary["label"],"candidate":b_summary["label"],"cases":cases,"selected_cases":selected_cases,"quality_only":quality_only,"allow_backend_change":allow_backend_change,"allow_sampling_change":allow_sampling_change,"baseline_backend":{"server":a_meta["server_sha256"],"libraries":a_meta["backend_libraries"]},"candidate_backend":{"server":b_meta["server_sha256"],"libraries":b_meta["backend_libraries"]},"scope":if quality_only {"matched model, inputs and prompt work; generated lengths may differ; no speed claims"} else {"matched model, inputs and evaluated work for selected cases; output agreement is not general quality parity; inspect hardware and thermal conditions separately"}}),
    )
}
fn main() -> Result<()> {
    let args = Args::parse();
    println!(
        "{}",
        serde_json::to_string_pretty(&compare(
            &args.baseline,
            &args.candidate,
            args.allow_backend_change,
            args.allow_sampling_change,
            args.quality_only,
            &args.selected_cases
        )?)?
    );
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn differing_answers_are_diagnostic_not_a_speedup() {
        let root = std::env::temp_dir().join(format!(
            "qwen-compare-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let meta = json!({"schema":2,"backend":"llama.cpp","server_sha256":"a","backend_libraries":{},"media_tools":{},"model_sha256":"b","mmproj_sha256":"c","repeats":1,"input":"fixture","settings":{"context":1024,"threads":6,"graphs":true,"flash_attention":true,"cache_prompt":false,"parallel":1,"ubatch":512,"batch":2048,"video_fps":4}});
        for (name, answer, tokens) in [
            ("left", "Two people", 3),
            ("right", "There are two people", 5),
        ] {
            let dir = root.join(name);
            fs::create_dir_all(&dir).unwrap();
            fs::write(dir.join("metadata.json"), meta.to_string()).unwrap();
            fs::write(dir.join("summary.json"), json!({"label":name}).to_string()).unwrap();
            let sample = json!({"case":"people","warmup":false,"request_sha256":"same","wall_ms":100.,"response":{"timings":{"cache_n":0,"prompt_n":100,"predicted_n":tokens},"choices":[{"message":{"content":answer}}]}});
            fs::write(dir.join("samples.jsonl"), format!("{sample}\n")).unwrap();
        }
        assert!(
            compare(
                &root.join("left"),
                &root.join("right"),
                false,
                false,
                false,
                &[]
            )
            .is_err()
        );
        let diagnostic = compare(
            &root.join("left"),
            &root.join("right"),
            false,
            false,
            true,
            &[],
        )
        .unwrap();
        assert_eq!(diagnostic["cases"][0]["sample_outputs_equal"], false);
        assert_eq!(diagnostic["cases"][0]["generated_work_equal"], false);
        assert!(diagnostic["cases"][0].get("speedup").is_none());
        assert!(
            compare(
                &root.join("left"),
                &root.join("right"),
                false,
                false,
                true,
                &["missing".into()]
            )
            .is_err()
        );
        let mut diagnostic_meta = meta.clone();
        diagnostic_meta["token_logprobs"] = json!(true);
        fs::write(root.join("left/metadata.json"), diagnostic_meta.to_string()).unwrap();
        assert!(
            compare(
                &root.join("left"),
                &root.join("left"),
                false,
                false,
                false,
                &[]
            )
            .unwrap_err()
            .to_string()
            .contains("diagnostic")
        );
        assert!(
            compare(
                &root.join("left"),
                &root.join("left"),
                false,
                false,
                true,
                &[]
            )
            .unwrap_err()
            .to_string()
            .contains("missing diagnostic")
        );
        let mut diagnostic_sample: Value = serde_json::from_str(
            fs::read_to_string(root.join("left/samples.jsonl"))
                .unwrap()
                .trim(),
        )
        .unwrap();
        diagnostic_sample["response"]["choices"][0]["logprobs"] = json!({"content":[
            {"id":32,"token":"A","logprob":-0.5,"top_logprobs":[{"id":32,"token":"A","logprob":-0.5}]}
        ]});
        fs::write(
            root.join("left/samples.jsonl"),
            format!("{diagnostic_sample}\n"),
        )
        .unwrap();
        let diagnostic = compare(
            &root.join("left"),
            &root.join("left"),
            false,
            false,
            true,
            &[],
        )
        .unwrap();
        assert_eq!(
            diagnostic["cases"][0]["token_differences"][0]["common_prefix_tokens"],
            1
        );
        assert!(diagnostic["cases"][0]["token_differences"][0]["first_difference_index"].is_null());
        fs::write(
            root.join("right/metadata.json"),
            diagnostic_meta.to_string(),
        )
        .unwrap();
        diagnostic_sample["response"]["choices"][0]["logprobs"]["content"][0]["id"] = json!(33);
        diagnostic_sample["response"]["choices"][0]["logprobs"]["content"][0]["token"] = json!("B");
        fs::write(
            root.join("right/samples.jsonl"),
            format!("{diagnostic_sample}\n"),
        )
        .unwrap();
        let changed = compare(
            &root.join("left"),
            &root.join("right"),
            false,
            false,
            true,
            &[],
        )
        .unwrap();
        assert_eq!(
            changed["cases"][0]["token_differences"][0]["first_difference_index"],
            0
        );
        assert_eq!(
            changed["cases"][0]["token_differences"][0]["candidate"]["token"],
            "B"
        );
        diagnostic_meta["token_logprobs"] = json!("invalid");
        fs::write(root.join("left/metadata.json"), diagnostic_meta.to_string()).unwrap();
        assert!(
            compare(
                &root.join("left"),
                &root.join("left"),
                false,
                false,
                true,
                &[]
            )
            .is_err()
        );
        fs::remove_dir_all(root).unwrap();
    }
    #[test]
    fn rejects_model_and_graph_policy_changes() {
        let a = json!({"schema":2,"backend":"llama.cpp","server_sha256":"a","backend_libraries":{"libllama.so":"d"},"media_tools":{},"model_sha256":"b","mmproj_sha256":"c","repeats":5,"input":"synthetic","settings":{"context":1024,"threads":6,"graphs":true,"flash_attention":true,"cache_prompt":false,"parallel":1,"ubatch":512,"video_fps":4}});
        let mut b = a.clone();
        b["settings"]["ubatch"] = json!(1024);
        assert!(same_contract(&a, &b, false, false).is_ok());
        b["model_sha256"] = json!("different");
        assert!(same_contract(&a, &b, false, false).is_err());
        assert!(same_contract(&a, &b, true, false).is_err());
        b = a.clone();
        b["settings"]["graphs"] = json!(false);
        assert!(same_contract(&a, &b, false, false).is_err());
        b = a.clone();
        b["backend_libraries"]["libllama.so"] = json!("changed");
        assert!(same_contract(&a, &b, false, false).is_err());
        let mut a = a;
        a["settings"]["batch"] = json!(2048);
        b["settings"]["batch"] = json!(2048);
        assert!(same_contract(&a, &b, true, false).is_ok());
        b["profiling"] = json!(true);
        assert!(same_contract(&a, &b, true, false).is_err());
        b = a.clone();
        b["settings"]["backend_sampling"] = json!(true);
        assert!(same_contract(&a, &b, false, false).is_err());
        assert!(same_contract(&a, &b, true, false).is_err());
        assert!(same_contract(&a, &b, false, true).is_ok());
        b["settings"]["backend_sampling"] = json!("invalid");
        assert!(same_contract(&a, &b, false, true).is_err());
        b["settings"]["backend_sampling"] = json!(true);
        b["model_sha256"] = json!("different");
        assert!(same_contract(&a, &b, false, true).is_err());
        b["model_sha256"] = a["model_sha256"].clone();
        b["backend_libraries"]["libllama.so"] = json!("changed");
        assert!(same_contract(&a, &b, false, true).is_err());
    }
}
