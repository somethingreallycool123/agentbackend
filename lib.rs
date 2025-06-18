// ✅ Or, if you defined it in lib.rs itself:
#[tauri::command]
fn greet(name: &str) -> String {
    format!("Hello, {}! You've been greeted from Rust!", name)
}

// ✅ YOUR capture command here too:
#[tauri::command]
fn capture_desktop(x: i32, y: i32, w: i32, h: i32) -> Result<String, String> {
    use scrap::{Capturer, Display};
    use image::{ImageBuffer, Rgba, DynamicImage};
    use std::{thread, time::Duration};
    use std::io::Cursor;
    use base64::engine::general_purpose;
    use base64::Engine;

    let display = Display::primary().map_err(|e| e.to_string())?;
    let mut capturer = Capturer::new(display).map_err(|e| e.to_string())?;

    let screen_w = capturer.width();
    let screen_h = capturer.height();

    thread::sleep(Duration::from_millis(16));
    let frame = loop {
        if let Ok(frame) = capturer.frame() {
            break frame;
        }
    };

    let mut img = ImageBuffer::<Rgba<u8>, Vec<u8>>::new(screen_w as u32, screen_h as u32);
    for (i, pixel) in img.pixels_mut().enumerate() {
        let idx = i * 4;
        *pixel = Rgba([frame[idx + 2], frame[idx + 1], frame[idx], 255]);
    }

    let sub = image::imageops::crop(&mut img, x.max(0) as u32, y.max(0) as u32, w as u32, h as u32).to_image();
    let small = image::imageops::resize(&sub, w as u32 / 2, h as u32 / 2, image::imageops::FilterType::Triangle);

    let dyn_img = DynamicImage::ImageRgba8(small);
    let mut png_bytes = Cursor::new(Vec::new());
    dyn_img.write_to(&mut png_bytes, image::ImageFormat::Png).map_err(|e| e.to_string())?;

    let bytes = png_bytes.into_inner();
    Ok(format!("data:image/png;base64,{}", general_purpose::STANDARD.encode(&bytes)))
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .invoke_handler(tauri::generate_handler![greet, capture_desktop])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
