import logging
import queue
import time

import cv2

from udp_signal_receiver import UDPSignalReceiver

logger = logging.getLogger(__name__)


def main():
    logger.info("Starting recording and processing service...")
    receiver = UDPSignalReceiver(debug_mode=True)
    receiver.start()

    logger.info("Service running. Program will stay active to receive signals")
    logger.info("Press Ctrl+C in this terminal to stop the service completely")

    try:
        while True:
            key_code = cv2.waitKey(1) & 0xFF

            if key_code == ord('q'):
                logger.info("'q' key pressed. Sending stop signal...")
                receiver._handle_stop_signal()

            try:
                frame_or_signal = receiver.display_queue.get_nowait()

                if frame_or_signal is None:
                    logger.info("Closing preview window...")
                    cv2.destroyWindow('Debug - Pose Detection')
                else:
                    cv2.imshow('Debug - Pose Detection', frame_or_signal)

            except queue.Empty:
                pass

            time.sleep(0.01)

    except KeyboardInterrupt:
        logger.info("Interrupt signal (Ctrl+C) received. Shutting down service...")
    finally:
        cv2.destroyAllWindows()
        logger.info("Service terminated")


if __name__ == "__main__":
    main()
