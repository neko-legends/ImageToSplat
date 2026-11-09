mod child_env;

use serde::Serialize;
use std::{
    path::PathBuf,
    process::{Child, Command, Stdio},
    sync::Mutex,
    thread,
    time::Duration,
};
use tauri::{Manager, State};

#[cfg(target_os = "windows")]
const CREATE_NO_WINDOW: u32 = 0x08000000;

struct AgentProcess {
    child: Mutex<Option<Child>>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct AgentLaunchStatus {
    url: String,
    port: u16,
    started: bool,
    message: String,
}

#[tauri::command]
fn start_agent_api(app: tauri::AppHandle, port: u16, state: State<'_, AgentProcess>) -> Result<AgentLaunchStatus, String> {
    let url = format!("http://127.0.0.1:{port}");
    if service_is_reachable(port) {
        return Ok(AgentLaunchStatus {
            url,
            port,
            started: false,
            message: "Local service is already running".to_string(),
        });
    }

    if !spawn_service(&app, port, &state)? {
        return Ok(AgentLaunchStatus {
            url,
            port,
            started: false,
            message: "Local service is starting".to_string(),
        });
    }

    for _ in 0..30 {
        if service_is_reachable(port) {
            return Ok(AgentLaunchStatus {
                url,
                port,
                started: true,
                message: "Local service started".to_string(),
            });
        }
        thread::sleep(Duration::from_millis(250));
    }

    Ok(AgentLaunchStatus {
        url,
        port,
        started: true,
        message: "Local service is still warming up".to_string(),
    })
}

/// Starts the Python service unless this app already has one running. Returns whether a new one was started.
fn spawn_service(app: &tauri::AppHandle, port: u16, state: &AgentProcess) -> Result<bool, String> {
    let mut guard = state
        .child
        .lock()
        .map_err(|_| "Unable to lock agent process state.".to_string())?;
    if let Some(child) = guard.as_mut() {
        if child.try_wait().map_err(|error| error.to_string())?.is_none() {
            return Ok(false);
        }
    }

    let project_root = project_root(app)?;
    let mut command = python_command(&project_root, port);
    child_env::strip_appimage_env(&mut command);
    let log = service_log_path();
    let (stdout, stderr) = match log.as_ref().and_then(|path| {
        std::fs::create_dir_all(path.parent()?).ok()?;
        let file = std::fs::File::create(path).ok()?;
        Some((file.try_clone().ok()?, file))
    }) {
        Some((out, err)) => (Stdio::from(out), Stdio::from(err)),
        None => (Stdio::null(), Stdio::null()),
    };
    #[cfg(target_os = "windows")]
    {
        use std::os::windows::process::CommandExt;
        command.creation_flags(CREATE_NO_WINDOW);
    }
    let child = command
        .current_dir(&project_root)
        .stdin(Stdio::null())
        .stdout(stdout)
        .stderr(stderr)
        .spawn()
        .map_err(|error| format!("Unable to start Python Agent API: {error}"))?;
    *guard = Some(child);
    Ok(true)
}

/// `--headless` / `--serve-agent-api`: run the local service without showing the window
/// (how Neko Legends Control Center starts an app's Agent API in the background).
fn headless_requested() -> bool {
    std::env::args().any(|arg| arg == "--headless" || arg == "--serve-agent-api")
}

fn cli_port() -> u16 {
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == "--agent-api-port" {
            if let Some(port) = args.next().and_then(|value| value.parse::<u16>().ok()) {
                return port;
            }
        } else if let Some(value) = arg.strip_prefix("--agent-api-port=") {
            if let Ok(port) = value.parse::<u16>() {
                return port;
            }
        }
    }
    17340
}

fn project_root(app: &tauri::AppHandle) -> Result<PathBuf, String> {
    if cfg!(debug_assertions) {
        return Ok(PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .ok_or_else(|| "Unable to resolve project root.".to_string())?
            .to_path_buf());
    }

    if let Ok(resource_dir) = app.path().resource_dir() {
        if resource_dir.join("run_gradio.py").is_file() {
            return Ok(resource_dir);
        }
    }

    std::env::current_exe()
        .map_err(|error| error.to_string())?
        .parent()
        .map(PathBuf::from)
        .ok_or_else(|| "Unable to resolve executable directory.".to_string())
}

/// Per-user data folder (Python environment, models, outputs, service log). `IMAGETOSPLAT_DATA_DIR` overrides it.
/// Only used on Linux; the Windows portable build keeps everything beside the exe.
fn data_dir() -> Option<PathBuf> {
    if let Some(dir) = std::env::var_os("IMAGETOSPLAT_DATA_DIR").filter(|v| !v.is_empty()) {
        return Some(PathBuf::from(dir));
    }
    let base = std::env::var_os("XDG_DATA_HOME")
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".local").join("share")))?;
    Some(base.join("NekoLegends").join("ImageToSplat"))
}

fn service_log_path() -> Option<PathBuf> {
    if cfg!(target_os = "linux") {
        data_dir().map(|dir| dir.join("service.log"))
    } else {
        None
    }
}

fn linux_service_script(project_root: &PathBuf) -> Option<PathBuf> {
    if !cfg!(target_os = "linux") {
        return None;
    }
    [project_root.join("linux-service.sh"), project_root.join("scripts").join("linux-service.sh")]
        .into_iter()
        .find(|path| path.is_file())
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ServiceSetupStatus {
    /// "starting", "installing", "failed" or "unknown"
    state: String,
    message: String,
    detail: String,
}

/// What the local service is doing before it answers: first-run package install progress, or why it failed.
#[tauri::command]
fn service_setup_status(state: State<'_, AgentProcess>) -> ServiceSetupStatus {
    let exited = state
        .child
        .lock()
        .ok()
        .and_then(|mut guard| guard.as_mut().and_then(|child| child.try_wait().ok().flatten()))
        .is_some();
    let text = service_log_path()
        .and_then(|path| std::fs::read_to_string(path).ok())
        .unwrap_or_default();
    let lines: Vec<&str> = text.lines().map(str::trim).filter(|line| !line.is_empty()).collect();
    let setup = lines.iter().rev().find(|line| line.starts_with("SETUP")).copied().unwrap_or("");
    let last: String = lines.last().copied().unwrap_or("").chars().take(200).collect();
    let (state, message) = if let Some(error) = setup.strip_prefix("SETUP-ERROR") {
        ("failed", error.trim().to_string())
    } else if exited {
        ("failed", format!("The local service stopped: {last}"))
    } else if setup.contains("Installing") || setup.contains("Creating") {
        ("installing", setup.trim_start_matches("SETUP").trim().to_string())
    } else if setup.is_empty() {
        ("unknown", String::new())
    } else {
        ("starting", setup.trim_start_matches("SETUP").trim().to_string())
    };
    ServiceSetupStatus {
        state: state.to_string(),
        message,
        detail: if last.starts_with("SETUP") { String::new() } else { last },
    }
}

fn python_command(project_root: &PathBuf, port: u16) -> Command {
    let port_arg = port.to_string();
    if let Some(script) = linux_service_script(project_root) {
        let mut command = Command::new("sh");
        command.arg(script).arg(port_arg);
        if let Some(dir) = data_dir() {
            command.env("IMAGETOSPLAT_DATA_DIR", dir);
        }
        // The service stops by itself if this app goes away without a clean exit.
        command.env("IMAGETOSPLAT_PARENT_PID", std::process::id().to_string());
        return command;
    }
    let bundled_exe = project_root.join("NekoSplatForge.exe");
    if bundled_exe.is_file() {
        let mut command = Command::new(bundled_exe);
        command
            .arg("--serve-agent-api")
            .arg("--agent-api-port")
            .arg(port_arg);
        return command;
    }

    let mut command = Command::new("python");
    command
        .arg("run_gradio.py")
        .arg("--serve-agent-api")
        .arg("--agent-api-port")
        .arg(port_arg);
    command
}

fn service_is_reachable(port: u16) -> bool {
    std::net::TcpStream::connect(("127.0.0.1", port)).is_ok()
}

pub fn run() {
    tauri::Builder::default()
        .manage(AgentProcess {
            child: Mutex::new(None),
        })
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_opener::init())
        .setup(|app| {
            if headless_requested() {
                if let Some(window) = app.get_webview_window("main") {
                    let _ = window.hide();
                }
                if let Err(error) = spawn_service(app.handle(), cli_port(), app.state::<AgentProcess>().inner()) {
                    eprintln!("Unable to start the local service: {error}");
                }
            }
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![start_agent_api, service_setup_status])
        .build(tauri::generate_context!())
        .expect("error while running ImageToSplat")
        .run(|app, event| {
            // The service this window started belongs to it: stop it on exit.
            if let tauri::RunEvent::Exit = event {
                if let Ok(mut guard) = app.state::<AgentProcess>().child.lock() {
                    if let Some(mut child) = guard.take() {
                        let _ = child.kill();
                        let _ = child.wait();
                    }
                }
            }
        });
}
