use tokio::sync::mpsc;
use tokio::time::{sleep, Duration, Instant};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

struct VideoPacket {
    camera_id: usize,
    frame_id: u64,
    size_bytes: usize,
}

#[tokio::main]
async fn main() {
    println!("=================================================");
    println!(" Rust High-Performance Async RTSP Stream Engine  ");
    println!(" Zero-Cost Abstractions + Tokio Async Runtime   ");
    println!("=================================================");

    let num_cameras = 16;
    let (tx, mut rx) = mpsc::channel::<VideoPacket>(1024);
    let total_packets = Arc::new(AtomicU64::new(0));

    // Spawn 16 Concurrent Tokio Camera Tasks
    for cam_id in 1..=num_cameras {
        let tx_clone = tx.clone();
        tokio::spawn(async move {
            let mut frame_id = 0u64;
            loop {
                sleep(Duration::from_millis(33)).await; // 30 FPS pacing
                frame_id += 1;
                let packet = VideoPacket {
                    camera_id: cam_id,
                    frame_id,
                    size_bytes: 6220800, // 1080p uncompressed BGR frame
                };
                if tx_clone.send(packet).await.is_err() {
                    break;
                }
            }
        });
    }

    drop(tx); // Drop initial sender

    let start_time = Instant::now();
    let counter = total_packets.clone();

    // Consumer Loop
    while let Some(packet) = rx.recv().await {
        let count = counter.fetch_add(1, Ordering::Relaxed) + 1;
        if count % 100 == 0 {
            let elapsed = start_time.elapsed().as_secs_f64();
            let fps = count as f64 / elapsed;
            println!(
                "[Rust Tokio] Processed {:4} packets | Aggregate Throughput: {:6.1} FPS | Camera #{:02} Frame #{:04}",
                count, fps, packet.camera_id, packet.frame_id
            );
        }
        if count >= 600 {
            break;
        }
    }

    let elapsed = start_time.elapsed().as_secs_f64();
    println!("\n=================================================");
    println!(" [SUCCESS] Rust Async Dispatcher Complete!");
    println!(" Total Packets Routed : {}", total_packets.load(Ordering::Relaxed));
    println!(" Aggregate Speed      : {:.1} FPS", 600.0 / elapsed);
    println!("=================================================");
}
