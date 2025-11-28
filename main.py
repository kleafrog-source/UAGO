import sys
import argparse

def main():
    parser = argparse.ArgumentParser(
        description="UAGO - Universal Adaptive Geometric Observer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --mode web          # Launch web interface (Streamlit)
  python main.py --mode local        # Launch local GUI (Tkinter)
  python main.py --mode web --port 8080  # Launch web interface on custom port
        """
    )

    parser.add_argument(
        '--mode',
        choices=['web', 'local'],
        default='web',
        help='Interface mode: web (Streamlit) or local (Tkinter)'
    )

    parser.add_argument(
        '--port',
        type=int,
        default=8501,
        help='Port for web interface (default: 8501)'
    )

    args = parser.parse_args()

    print("=" * 60)
    print("UAGO - Universal Adaptive Geometric Observer")
    print("Version 1.0")
    print("=" * 60)
    print()

    if args.mode == 'web':
        print(f"Launching web interface on port {args.port}...")
        print("Opening Streamlit application...")
        print()

        import subprocess
        subprocess.run([
            sys.executable,
            "-m",
            "streamlit",
            "run",
            "app_streamlit.py",
            "--server.port",
            str(args.port)
        ])

    elif args.mode == 'local':
        print("Launching local GUI (Tkinter)...")
        print()

        from app_tkinter import main as tkinter_main
        tkinter_main()

if __name__ == "__main__":
    main()
try:
    result = subprocess.run([...], check=True, timeout=None)  # No timeout for manual stop
except KeyboardInterrupt:
    print("Streamlit stopped by user.")
    sys.exit(0)
except subprocess.TimeoutExpired:
    print("Streamlit timed out.")