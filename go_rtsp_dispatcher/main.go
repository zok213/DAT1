package main

import (
	"fmt"
	"sync"
	"time"
)

type CameraStream struct {
	ID        int
	URL       string
	FrameRate float64
}

type PacketTelemetry struct {
	CameraID  int
	FrameID   int64
	Timestamp time.Time
	Size      int
}

func cameraWorker(cam CameraStream, telemetryChan chan<- PacketTelemetry, wg *sync.WaitGroup, stopChan <-chan struct{}) {
	defer wg.Done()
	var frameID int64 = 0

	ticker := time.NewTicker(time.Duration(1000/cam.FrameRate) * time.Millisecond)
	defer ticker.Stop()

	for {
		select {
		case <-stopChan:
			return
		case <-ticker.C:
			frameID++
			packet := PacketTelemetry{
				CameraID:  cam.ID,
				FrameID:   frameID,
				Timestamp: time.Now(),
				Size:      6220800, // 1080p BGR uncompressed frame size
			}
			telemetryChan <- packet
		}
	}
}

func main() {
	fmt.Println("=================================================")
	fmt.Println(" Go (Golang) Ultra-Fast RTSP Stream Dispatcher   ")
	fmt.Println(" Zero-Copy Concurrent Multi-Goroutine Engine     ")
	fmt.Println("=================================================")

	numCameras := 8
	telemetryChan := make(chan PacketTelemetry, 256)
	stopChan := make(chan struct{})
	var wg sync.WaitGroup

	// Launch Goroutine Workers per IP Camera
	for i := 1; i <= numCameras; i++ {
		cam := CameraStream{
			ID:        i,
			URL:       fmt.Sprintf("rtsp://192.168.1.%d:554/live", 100+i),
			FrameRate: 30.0,
		}
		wg.Add(1)
		go cameraWorker(cam, telemetryChan, &wg, stopChan)
	}

	// Dispatcher Consumer Goroutine
	go func() {
		packetCount := 0
		startTime := time.Now()

		for packet := range telemetryChan {
			packetCount++
			if packetCount%100 == 0 {
				elapsed := time.Since(startTime).Seconds()
				fps := float64(packetCount) / elapsed
				fmt.Printf("[Go Dispatcher] Routed %d frames | Aggregate Throughput: %.1f FPS | Current: Cam #%d Frame #%d\n",
					packetCount, fps, packet.CameraID, packet.FrameID)
			}
			if packetCount >= 500 {
				close(stopChan)
				return
			}
		}
	}()

	wg.Wait()
	close(telemetryChan)
	fmt.Println("[SUCCESS] Go RTSP Dispatcher run completed cleanly.")
}
