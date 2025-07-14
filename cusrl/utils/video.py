__all__ = ["VideoWriter"]


class VideoWriter:
    def __init__(self, path: str, resolution: tuple[int, int], fps: float):
        try:
            import cv2
        except ImportError as error:
            raise ImportError("Please run 'pip install opencv-python' to use VideoRecorder.") from error

        self.path = path
        self.out = cv2.VideoWriter(path, 0, fps, resolution)

    def write(self, frame):
        self.out.write(frame)

    def release(self):
        self.out.release()

    def __del__(self):
        self.release()
