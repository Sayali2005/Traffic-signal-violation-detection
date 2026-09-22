"""
Command Line Interface (CLI) for Traffic Signal Violation Detection.
Processes video files, outputs annotated results, saves violation snapshots,
and exports comprehensive statistical reports.
"""

import os
import sys
import argparse
import time
import json
import csv
import cv2
from pipeline import TrafficViolationPipeline

def parse_args():
    parser = argparse.ArgumentParser(description="Traffic Signal Violation Detection System (IJARET 2021)")
    parser.add_argument("--video", type=str, default="videos/traffic_ip_camera.mp4", help="Path to input video")
    parser.add_argument("--output", type=str, default="output_annotated.mp4", help="Path to output annotated video (optional)")
    parser.add_argument("--stop-line", type=str, default="180,500,1120,500", help="Stop line coordinates 'x1,y1,x2,y2'")
    parser.add_argument("--speed-limit", type=float, default=50.0, help="Speed limit in km/h")
    parser.add_argument("--meters-per-pixel", type=float, default=0.045, help="Meters per pixel calibration factor")
    parser.add_argument("--signal-mode", type=str, default="AUTO_CYCLE", choices=["AUTO_CYCLE", "MANUAL", "COLOR_ROI"], help="Traffic signal mode")
    parser.add_argument("--initial-state", type=str, default="GREEN", choices=["GREEN", "YELLOW", "RED"], help="Initial traffic signal state")
    parser.add_argument("--green-time", type=float, default=10.0, help="Green light duration (s)")
    parser.add_argument("--yellow-time", type=float, default=3.0, help="Yellow light duration (s)")
    parser.add_argument("--red-time", type=float, default=10.0, help="Red light duration (s)")
    parser.add_argument("--detect-interval", type=int, default=2, help="Run YOLO detection every N frames (1 = every frame, 2-3 = fast CPU mode)")
    parser.add_argument("--imgsz", type=int, default=480, help="YOLO inference image size (e.g. 384, 480, 640)")
    parser.add_argument("--max-frames", type=int, default=0, help="Max frames to process (0 = all)")
    parser.add_argument("--show", action="store_true", help="Display live OpenCV window")
    parser.add_argument("--keep-snapshots", action="store_true", help="Preserve previous snapshots in ./violations/")
    return parser.parse_args()

def main():
    args = parse_args()

    if not os.path.exists(args.video):
        print(f"[Error] Video file not found: {args.video}")
        sys.exit(1)

    # Clean previous run snapshots if not explicitly preserved
    violations_dir = "./violations"
    if not args.keep_snapshots and os.path.exists(violations_dir):
        for f in os.listdir(violations_dir):
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                try:
                    os.remove(os.path.join(violations_dir, f))
                except Exception:
                    pass

    # Parse stop line
    try:
        coords = [int(v.strip()) for v in args.stop_line.split(",")]
        assert len(coords) == 4
        stop_line = ((coords[0], coords[1]), (coords[2], coords[3]))
    except Exception:
        print(f"[Warning] Invalid stop line format '{args.stop_line}'. Defaulting to ((180, 500), (1120, 500))")
        stop_line = ((180, 500), (1120, 500))

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print("=" * 65)
    print(" TRAFFIC SIGNAL VIOLATION DETECTION SYSTEM")
    print(" Paper: Anand et al., IJARET 2021")
    print("=" * 65)
    print(f"Input Video       : {args.video} ({width}x{height} @ {fps:.1f} FPS, {total_frames} frames)")
    print(f"Stop Line         : {stop_line[0]} -> {stop_line[1]}")
    print(f"Speed Limit       : {args.speed_limit} km/h")
    print(f"Signal Mode       : {args.signal_mode}")
    print(f"Detect Interval   : Every {args.detect_interval} frames (imgsz={args.imgsz})")
    print(f"Saving Snapshots  : ./violations/")
    print("=" * 65)

    # Initialize Pipeline
    pipeline = TrafficViolationPipeline(
        fps=fps,
        stop_line=stop_line,
        speed_limit_kmh=args.speed_limit,
        meters_per_pixel=args.meters_per_pixel,
        signal_mode=args.signal_mode,
        initial_state=args.initial_state,
        green_duration=args.green_time,
        yellow_duration=args.yellow_time,
        red_duration=args.red_time,
        detect_interval=args.detect_interval,
        imgsz=args.imgsz
    )

    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    processed_count = 0
    start_time = time.time()

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            processed_count += 1
            if args.max_frames > 0 and processed_count > args.max_frames:
                break

            annotated_frame, stats, new_violations = pipeline.process_frame(frame)

            if writer:
                writer.write(annotated_frame)

            for viol in new_violations:
                print(f" >> [ALERT] Frame {viol['frame_idx']} | ID #{viol['track_id']} ({viol['class_name']}) | "
                      f"Speed: {viol['speed_kmh']} km/h | Violation: {viol['violation_type']} | Light: {viol['signal_state']}")

            if processed_count % 30 == 0 or processed_count == total_frames:
                elapsed = time.time() - start_time
                fps_proc = processed_count / max(0.001, elapsed)
                pct = (processed_count / max(1, total_frames)) * 100
                print(f"Progress: [{processed_count}/{total_frames}] {pct:.1f}% | Processing Speed: {fps_proc:.1f} FPS | "
                      f"Vehicles: {stats['total_vehicles']} | Violations: {stats['total_violations']}", end="\r")

            if args.show:
                cv2.imshow("Traffic Signal Violation Detection", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    finally:
        cap.release()
        if writer:
            writer.release()
        if args.show:
            cv2.destroyAllWindows()

    total_time = time.time() - start_time
    all_violations = pipeline.violation_detector.violations

    # Export report to CSV
    csv_path = "violations_report.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer_csv = csv.writer(f)
        writer_csv.writerow(["Violation_ID", "Vehicle_ID", "Class", "Speed_kmh", "Violation_Type", "Signal_State", "Frame", "Timestamp", "Evidence_Snapshot"])
        for v in all_violations:
            writer_csv.writerow([
                v['violation_id'],
                v['track_id'],
                v['class_name'],
                v['speed_kmh'],
                v['violation_type'],
                v['signal_state'],
                v['frame_idx'],
                v['timestamp'],
                v['snapshot_file']
            ])

    # Export report to JSON
    json_path = "violations_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            'video': args.video,
            'processed_frames': processed_count,
            'processing_time_sec': round(total_time, 2),
            'average_fps': round(processed_count / max(0.001, total_time), 2),
            'total_vehicles_counted': pipeline.tracker.total_counted,
            'total_violations_recorded': len(all_violations),
            'signal_jump_count': sum(1 for v in all_violations if "Signal Jump" in v['violation_type']),
            'speed_violation_count': sum(1 for v in all_violations if "Speed Violation" in v['violation_type']),
            'violations': all_violations
        }, f, indent=2)

    print("\n" + "=" * 65)
    print(" EXECUTION COMPLETE - PERFORMANCE ANALYSIS SUMMARY")
    print("=" * 65)
    print(f"Frames Processed      : {processed_count}")
    print(f"Total Execution Time  : {total_time:.2f} seconds ({processed_count / max(0.001, total_time):.1f} FPS)")
    print(f"Total Vehicles Counted: {pipeline.tracker.total_counted}")
    print(f"Total Violations      : {len(all_violations)}")
    print(f"  - Signal Jumps      : {sum(1 for v in all_violations if 'Signal Jump' in v['violation_type'])}")
    print(f"  - Speeding Exceeded : {sum(1 for v in all_violations if 'Speed Violation' in v['violation_type'])}")
    print(f"Violations CSV Report : {csv_path}")
    print(f"Violations JSON Report: {json_path}")
    if args.output:
        print(f"Annotated Video Output: {args.output}")
    print("=" * 65)

if __name__ == "__main__":
    main()
