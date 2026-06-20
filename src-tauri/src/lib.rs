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

    {
        let mut guard = state
            .child
            .lock()
            .map_err(|_| "Unable to lock agent process state.".to_string())?;
        if let Some(child) = guard.as_mut() {
            if child.try_wait().map_err(|error| error.to_string())?.is_none() {
                return Ok(AgentLaunchStatus {
                    url,
                    port,
                    started: false,
                    message: "Local service is starting".to_string(),
                });
            }
        }

        let project_root = project_root(&app)?;
        let mut command = python_command(&project_root, port);
        #[cfg(target_os = "windows")]
        {
            use std::os::windows::process::CommandExt;
            command.creation_flags(CREATE_NO_WINDOW);
        }
        let child = command
            .current_dir(&project_root)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|error| format!("Unable to start Python Agent API: {error}"))?;
        *guard = Some(child);
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

fn python_command(project_root: &PathBuf, port: u16) -> Command {
    let port_arg = port.to_string();
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
        .invoke_handler(tauri::generate_handler![start_agent_api])
        .run(tauri::generate_context!())
        .expect("error while running ImageToSplat");
}
