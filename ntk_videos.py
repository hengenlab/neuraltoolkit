import cv2
import os


class NTKVideos:
    '''ntk video class, interface to opencv

    get video attributes
    Example:
    videofilename = '/home/user/e3v810a-20190307T0740-0840.mp4'
    lstream = 0
    lstream is 1 is video is streaming or 0 if video is already saved
    v  = NTKVideos(videofilename, lstream)
    v contains length, width, height information from video

    play video, please press q to exit
    v.play_video()

    '''

    def __init__(self, name, lstream):
        self.cap = cv2.VideoCapture(name)

        # Check if camera opened successfully
        if self.cap.isOpened() is True:
            if lstream == 0:
                self.length = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
            self.width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            self.height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            codec = int(self.cap.get(cv2.CAP_PROP_FOURCC))
            self.codec = [chr((codec >> 8 * i) & 0xFF) for i in range(4)]
            self.brightness = self.cap.get(cv2.CAP_PROP_BRIGHTNESS)
            self.contrast = self.cap.get(cv2.CAP_PROP_CONTRAST)

        else:
            print("Error opening video stream or file")

    def play_video(self):
            print("Please press q to exit")
            while True:
                # Capture frame-by-frame
                self.ret, self.frame = self.cap.read()

                if self.ret is True:
                    cv2.imshow('video', self.frame)
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        break
            self.cap.release()
            cv2.destroyAllWindows()


if __name__ == '__main__':
    videofilename = 'e3v810a-20190307T0740-0840.mp4'
    # videofilename = 0
    lstream = 0
    v = NTKVideos(videofilename, lstream)
