use anyhow::{Context, Result, bail, ensure};
use base64::{Engine, engine::general_purpose::STANDARD};
use clap::Parser;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    fs::{self, File, OpenOptions},
    io::{Read, Write},
    net::TcpStream,
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    thread,
    time::{Duration, Instant},
};

/// Measure a pinned native Qwen backend using deterministic synthetic visual inputs.
#[derive(Parser, Debug)]
struct Args {
    #[arg(long)]
    server: PathBuf,
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    mmproj: PathBuf,
    #[arg(long)]
    out: PathBuf,
    #[arg(long)]
    label: String,
    #[arg(long, default_value_t = 8185)]
    port: u16,
    #[arg(long, default_value_t = 5)]
    repeats: usize,
    #[arg(long, default_value_t = 512)]
    ubatch: usize,
    #[arg(long, default_value_t = 2048)]
    batch: usize,
    #[arg(long, default_value_t = 16384)]
    context: usize,
    #[arg(long, default_value_t = 6)]
    threads: usize,
    /// Diagnostic ablation only; never label graphs-off as the strong baseline.
    #[arg(long)]
    disable_graphs: bool,
    /// Comma-separated counts of separately encoded synthetic images. Not native video.
    #[arg(long, value_delimiter = ',', default_value = "0,1,8,48")]
    images: Vec<usize>,
    /// Add a native video case using this public/synthetic MP4 fixture.
    #[arg(long)]
    video: Option<PathBuf>,
    #[arg(long, default_value_t = 4)]
    video_fps: u32,
}

struct Server(Child);
impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

fn hash_file(path: &Path) -> Result<String> {
    let mut file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut hash = Sha256::new();
    let mut buf = vec![0; 1024 * 1024];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hash.update(&buf[..n]);
    }
    Ok(format!("{:x}", hash.finalize()))
}
fn hash_bytes(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn backend_libraries(server: &Path) -> Result<Value> {
    let parent = server
        .canonicalize()?
        .parent()
        .context("server has no parent")?
        .to_path_buf();
    let mut hashes = std::collections::BTreeMap::new();
    for entry in fs::read_dir(parent)? {
        let entry = entry?;
        let name = entry.file_name().to_string_lossy().into_owned();
        if entry.path().is_file()
            && (name.ends_with(".so") || name.contains(".so.") || name.ends_with(".dylib"))
        {
            hashes.insert(name, hash_file(&entry.path())?);
        }
    }
    Ok(serde_json::to_value(hashes)?)
}

fn media_tools() -> Result<Value> {
    let path = std::env::var_os("PATH").context("PATH is missing")?;
    let mut tools = serde_json::Map::new();
    for name in ["ffmpeg", "ffprobe"] {
        let file = std::env::split_paths(&path)
            .map(|dir| dir.join(name))
            .find(|file| file.is_file())
            .with_context(|| format!("{name} is required for native video"))?;
        let version = Command::new(&file).arg("-version").output()?;
        ensure!(version.status.success(), "{name} -version failed");
        tools.insert(name.into(), json!({"sha256":hash_file(&file)?,"version":String::from_utf8_lossy(&version.stdout).lines().next()}));
    }
    Ok(Value::Object(tools))
}

fn image(frame: usize) -> Result<Vec<u8>> {
    let size = 448usize;
    let mut pixels = vec![240u8; size * size * 3];
    let left = 24 + frame * 7 % 320;
    for y in 160..240 {
        for x in left..left + 64 {
            let i = (y * size + x) * 3;
            pixels[i..i + 3].copy_from_slice(&[220, 30, 30]);
        }
    }
    let mut bytes = Vec::new();
    {
        let mut encoder = png::Encoder::new(&mut bytes, size as u32, size as u32);
        encoder.set_color(png::ColorType::Rgb);
        encoder.set_depth(png::BitDepth::Eight);
        encoder.write_header()?.write_image_data(&pixels)?;
    }
    Ok(bytes)
}
fn request(images: usize) -> Result<Value> {
    let mut content = Vec::new();
    for i in 0..images {
        content.push(json!({"type":"text", "text":format!("Frame {i}: ")}));
        content.push(json!({"type":"image_url","image_url":{"url":format!("data:image/png;base64,{}",STANDARD.encode(image(i)?))}}));
    }
    let prompt = if images == 0 {
        "Write a detailed explanation of how a bicycle works, covering wheels, chain, gears, brakes and steering."
    } else {
        "Is a red rectangle visible? Answer YES or NO."
    };
    content.push(json!({"type":"text","text":prompt}));
    Ok(
        json!({"messages":[{"role":"user","content":content}], "temperature":0,
        "seed":42,"max_tokens":if images == 0 {128} else {4},"stream":false,
        "cache_prompt":false,"return_tokens":true,"timings_per_token":true,
        "chat_template_kwargs":{"enable_thinking":false}}),
    )
}
fn validate_response(response: &Value) -> Result<()> {
    ensure!(
        response.get("error").is_none(),
        "backend returned error: {response}"
    );
    let timings = &response["timings"];
    ensure!(
        timings["cache_n"].as_u64() == Some(0),
        "cached or missing cache count: {timings}"
    );
    ensure!(
        timings["prompt_n"].as_u64().unwrap_or(0) > 0,
        "no prompt evaluated"
    );
    ensure!(
        timings["predicted_n"].as_u64().unwrap_or(0) > 0,
        "no tokens generated"
    );
    ensure!(
        response["choices"][0]["message"]["content"]
            .as_str()
            .is_some_and(|s| !s.is_empty()),
        "empty completion"
    );
    Ok(())
}

fn video_request(bytes: &[u8]) -> Result<Value> {
    ensure!(!bytes.is_empty(), "video fixture is empty");
    let mut body = request(1)?;
    body["messages"][0]["content"] = json!([
        {"type":"input_video","input_video":{"data":STANDARD.encode(bytes)}},
        {"type":"text","text":"Is a red rectangle visible? Answer YES or NO."}
    ]);
    Ok(body)
}
fn percentile(values: &mut [f64], q: f64) -> f64 {
    values.sort_by(f64::total_cmp);
    let index = (values.len() - 1) as f64 * q;
    let lo = index.floor() as usize;
    let hi = index.ceil() as usize;
    values[lo] + (values[hi] - values[lo]) * index.fract()
}
fn write_json(path: &Path, value: &Value) -> Result<()> {
    let mut f = OpenOptions::new().write(true).create_new(true).open(path)?;
    serde_json::to_writer_pretty(&mut f, value)?;
    f.write_all(b"\n")?;
    Ok(())
}
fn gpu_snapshot() -> Result<String> {
    let result = Command::new("nvidia-smi").args(["--query-gpu=name,memory.total,memory.used,utilization.gpu,driver_version,power.draw,temperature.gpu", "--format=csv"]).output()?;
    ensure!(result.status.success(), "nvidia-smi failed");
    Ok(String::from_utf8(result.stdout)?)
}
fn run(args: Args) -> Result<()> {
    ensure!(args.video_fps > 0, "video FPS must be positive");
    ensure!(
        args.repeats >= 2,
        "at least two measured repetitions required"
    );
    ensure!(
        !args.images.is_empty() && args.images.iter().all(|&n| n <= 48),
        "image counts must be in 0..=48"
    );
    ensure!(
        args.ubatch > 0 && args.batch >= args.ubatch && args.threads > 0 && args.context > 0,
        "invalid batch/context/thread settings"
    );
    let address = format!("127.0.0.1:{}", args.port);
    ensure!(
        TcpStream::connect(&address).is_err(),
        "benchmark port is already in use"
    );
    fs::create_dir(&args.out).context("output directory must not already exist")?;
    let before = gpu_snapshot()?;
    let args_native = vec![
        "--host".into(),
        "127.0.0.1".into(),
        "--port".into(),
        args.port.to_string(),
        "-m".into(),
        args.model.to_string_lossy().into_owned(),
        "--mmproj".into(),
        args.mmproj.to_string_lossy().into_owned(),
        "-ngl".into(),
        "999".into(),
        "-c".into(),
        args.context.to_string(),
        "-b".into(),
        args.batch.to_string(),
        "-ub".into(),
        args.ubatch.to_string(),
        "-t".into(),
        args.threads.to_string(),
        "-np".into(),
        "1".into(),
        "--cache-ram".into(),
        "0".into(),
        "--no-cache-prompt".into(),
        "--flash-attn".into(),
        "on".into(),
        "--jinja".into(),
        "--no-webui".into(),
        "--video-fps".into(),
        args.video_fps.to_string(),
    ];
    let version = Command::new(&args.server).arg("--version").output()?;
    ensure!(
        version.status.success(),
        "server --version failed: {}",
        String::from_utf8_lossy(&version.stderr)
    );
    let video = args.video.as_ref().map(fs::read).transpose()?;
    let metadata = json!({"schema":2,"label":args.label,"backend":"llama.cpp", "server_sha256":hash_file(&args.server)?,
        "backend_libraries":backend_libraries(&args.server)?,
        "media_tools":if args.video.is_some() { media_tools()? } else {json!({})},
        "server_version_stdout":String::from_utf8_lossy(&version.stdout), "server_version_stderr":String::from_utf8_lossy(&version.stderr),
        "model_sha256":hash_file(&args.model)?,"mmproj_sha256":hash_file(&args.mmproj)?,"gpu_before":before,
        "settings":{"ubatch":args.ubatch,"batch":args.batch,"context":args.context,"threads":args.threads,"graphs":!args.disable_graphs,"flash_attention":true,"cache_prompt":false,"parallel":1,"video_fps":args.video_fps},
        "repeats":args.repeats,"warmups_per_case":1,"video_sha256":video.as_ref().map(|v| hash_bytes(v)),
        "input":"synthetic 448x448 PNG image sequences; optional separately labeled native video", "timing":"warm HTTP request wall time; excludes client fixture encoding, process startup, model loading; includes server media processing"});
    write_json(&args.out.join("metadata.json"), &metadata)?;
    let log = File::create(args.out.join("server.log"))?;
    let mut command = Command::new(&args.server);
    command
        .args(&args_native)
        .stdout(Stdio::from(log.try_clone()?))
        .stderr(Stdio::from(log));
    // Do not let the caller's graph override silently invalidate the declared experiment.
    command.env_remove("GGML_CUDA_DISABLE_GRAPHS");
    if args.disable_graphs {
        command.env("GGML_CUDA_DISABLE_GRAPHS", "1");
    }
    let mut server = Server(command.spawn()?);
    let agent = ureq::AgentBuilder::new()
        .timeout(Duration::from_secs(300))
        .build();
    let base = format!("http://{address}");
    let loading = Instant::now();
    loop {
        if agent
            .get(&format!("{base}/health"))
            .timeout(Duration::from_secs(1))
            .call()
            .is_ok()
        {
            break;
        }
        if let Some(status) = server.0.try_wait()? {
            bail!("server exited {status}; inspect server.log");
        }
        ensure!(
            loading.elapsed() < Duration::from_secs(180),
            "server readiness timeout"
        );
        thread::sleep(Duration::from_millis(200));
    }
    let mut raw = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(args.out.join("samples.jsonl"))?;
    let mut summaries = Vec::new();
    let mut cases = Vec::new();
    for count in &args.images {
        cases.push((format!("images-{count}"), request(*count)?));
    }
    if let Some(bytes) = video {
        cases.push(("native-video".to_string(), video_request(&bytes)?));
    }
    for (case, request) in cases {
        let body = serde_json::to_vec(&request)?;
        let mut times = Vec::new();
        let mut outputs = Vec::new();
        let mut decode = Vec::new();
        for repetition in 0..=args.repeats {
            let start = Instant::now();
            let response: Value = agent
                .post(&format!("{base}/v1/chat/completions"))
                .set("Content-Type", "application/json")
                .send_bytes(&body)?
                .into_json()?;
            let elapsed = start.elapsed().as_secs_f64() * 1000.;
            validate_response(&response)?;
            let sample = json!({"case":case,"warmup":repetition==0,"repetition":repetition,"request_sha256":hash_bytes(&body),"wall_ms":elapsed,"response":response,"gpu_after":gpu_snapshot()?});
            serde_json::to_writer(&mut raw, &sample)?;
            raw.write_all(b"\n")?;
            raw.flush()?;
            if repetition != 0 {
                times.push(elapsed);
                outputs.push(response["choices"][0]["message"]["content"].clone());
                decode.push(
                    response["timings"]["predicted_per_second"]
                        .as_f64()
                        .context("missing decode throughput")?,
                );
            }
            eprintln!("case={case} repetition={repetition} wall_ms={elapsed:.2}");
        }
        let mean = times.iter().sum::<f64>() / times.len() as f64;
        summaries.push(json!({"case":case,"request_sha256":hash_bytes(&body),"mean_wall_ms":mean,"p50_wall_ms":percentile(&mut times,0.5),"p95_wall_ms":percentile(&mut times,0.95),"serial_requests_per_minute":60000./mean,"mean_decode_tokens_per_second":decode.iter().sum::<f64>()/decode.len() as f64,"outputs_identical":outputs.windows(2).all(|w| w[0]==w[1]),"outputs":outputs}));
    }
    write_json(
        &args.out.join("summary.json"),
        &json!({"label":args.label,"cases":summaries,"quality":"synthetic smoke only; no real-world accuracy claim","gpu_after":gpu_snapshot()?}),
    )?;
    Ok(())
}
fn main() -> Result<()> {
    run(Args::parse())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rejects_cached_or_missing_measurements() {
        let valid = json!({"timings":{"cache_n":0,"prompt_n":20,"predicted_n":1},"choices":[{"message":{"content":"YES"}}]});
        assert!(validate_response(&valid).is_ok());
        let mut cached = valid.clone();
        cached["timings"]["cache_n"] = json!(20);
        assert!(validate_response(&cached).is_err());
        let mut missing = valid;
        missing.as_object_mut().unwrap().remove("timings");
        assert!(validate_response(&missing).is_err());
    }
    #[test]
    fn fixture_and_request_contract() {
        assert_eq!(image(0).unwrap(), image(0).unwrap());
        assert_ne!(image(0).unwrap(), image(1).unwrap());
        let r = request(8).unwrap();
        assert_eq!(
            r["messages"][0]["content"]
                .as_array()
                .unwrap()
                .iter()
                .filter(|v| v["type"] == "image_url")
                .count(),
            8
        );
        assert_eq!(r["cache_prompt"], false);
        assert_eq!(r["chat_template_kwargs"]["enable_thinking"], false);
        let decoder = png::Decoder::new(std::io::Cursor::new(image(0).unwrap()));
        let reader = decoder.read_info().unwrap();
        assert_eq!((reader.info().width, reader.info().height), (448, 448));
        let video = video_request(b"fixture").unwrap();
        assert_eq!(video["messages"][0]["content"][0]["type"], "input_video");
        assert_eq!(
            video["messages"][0]["content"][0]["input_video"]["data"],
            STANDARD.encode(b"fixture")
        );
        assert!(video_request(&[]).is_err());
    }
    #[test]
    fn summary_quantiles_use_interpolation() {
        assert_eq!(percentile(&mut [30., 10., 20.], 0.5), 20.);
        assert_eq!(percentile(&mut [30., 10., 20.], 0.95), 29.);
    }
}
